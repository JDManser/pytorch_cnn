# @author: The Plant Pathologists (John Manser, Ben Dorevitch, Matthew Crick, Keira Sherri, Alexander Small, and Ryley Brohier)
# FARMERS: Fully Automated disease Recognition by Machine learning Employing Remote Sensing

import os
import numpy as np
import spectral as sp
from spectral import *
from skimage.filters import threshold_otsu
from skimage.morphology import *
from sklearn.cluster import KMeans
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import torch

class Segmentor:
    
    @staticmethod
    def __get_bands_in_range(hsi_image, wavelength_start, wavelength_end, bands):
        images = []
        for id, band in enumerate(bands):

            if band >= wavelength_start and band <= wavelength_end:
                images.append(hsi_image[:,:,id])
        return images

    @staticmethod
    # Get the average reflectance for a HSI between two wavelengths.
    def get_average_image_for_range(hsi_image, wavelength_start, wavelength_end, bands):

        # Get all images which fall inside this wavelength range.
        image_array = Segmentor.__get_bands_in_range(hsi_image, wavelength_start, wavelength_end, bands)

        # Calculate the average reflectance of these images by doing an element-wise mean operation.
        return np.mean(image_array, axis=0)

    # Get the NDVI of a HSI using ranges to determine the start and end of the red and VNIR wavelengths.
    @staticmethod
    def __get_ndvi_using_ranges(image, bands, red_wavelength_start, red_wavelength_end, vnir_wavelength_start, vnir_wavelength_end):
        mean_red_intensity =  Segmentor.get_average_image_for_range(image, red_wavelength_start,  red_wavelength_end,  bands)
        mean_vnir_intensity = Segmentor.get_average_image_for_range(image, vnir_wavelength_start, vnir_wavelength_end, bands)

        return (mean_red_intensity - mean_vnir_intensity) / (mean_red_intensity + mean_vnir_intensity)

    @staticmethod
    def __denoise_segmentation_mask(segmentation_mask):
        eroded_image = binary_opening(segmentation_mask) # Get rid of tiny white spots ('salt')
        eroded_image = remove_small_objects(eroded_image, 256) # Get rid of tiny white spots ('salt')

        # Dilate the image to remove small holes
        for i in range(3):
            eroded_image = dilation(eroded_image)

        eroded_image = remove_small_objects(eroded_image, 256) # Get rid of tiny white spots ('salt')
        
        #eroded_image = isotropic_erosion(eroded_image, 3) # Erode awway most of the noise
        eroded_image = isotropic_opening(eroded_image,2)
        eroded_image = binary_closing(eroded_image)
        eroded_image = erosion(eroded_image)

        return eroded_image

    @staticmethod
    def __apply_segmentation_mask(hsi_image, segmentation_mask):
        hsi_image = np.array(hsi_image)
        
        new_image = np.copy(hsi_image)
        
        new_image[~segmentation_mask] = 0 # Set all pixels in the image which are segmented to 0

        return new_image

    @staticmethod
    def segment_image(hsi_image, bands, red_wavelength_start = 710, red_wavelength_end = 740, vnir_wavelength_start = 650, vnir_wavelength_end = 680):
        # Get the NDVI of the image
        ndvi = Segmentor.__get_ndvi_using_ranges(hsi_image, bands, red_wavelength_start, red_wavelength_end, vnir_wavelength_start, vnir_wavelength_end)

        threshold = threshold_otsu(ndvi)

        binary_segment = np.squeeze(ndvi > threshold)

        eroded_image = Segmentor.__denoise_segmentation_mask(binary_segment)

        image_segment_mask = eroded_image

        segmented_image = Segmentor.__apply_segmentation_mask(hsi_image, image_segment_mask)

        return segmented_image

class Chunking:
    # Returns how many background (black) pixels are in a chunk as a percentage from 0 to 1
    @staticmethod
    def get_plant_ratio_in_chunk(chunk):
        chunk = np.array(chunk)

        pixels = chunk.flatten()

        # How many pixels in the chunk have an intensity of 0?
        # (We will count them as background pixels)
        background_pixel_count = (pixels==0).sum()

        return background_pixel_count / len(pixels)

    # Split a hyperspectral image into an array of chunks.
    # hyperspectral_image: A 3D NumPy array or loaded hyperspectral image object.
    # chunk_size: The height and width of chunks of the hyperspectral image. (i.e., 32)
    # enforce_chunk_size: Whether to force chunks to be of size chunk_size when the dimensions of the image are not a factor of chunk_size.
    #                     If true, chunks at the edges of the image will be padded with zeroes if they exceed the image size.
    @staticmethod
    def slice_hsi_into_chunks(hyperspectral_image, chunk_size : int, enforce_chunk_size : bool = False):

        chunks = []
        # Get the width and height for the image    
        image_size = np.array(list(hyperspectral_image.shape[:-1]))
        number_of_bands = int(hyperspectral_image.shape[-1])

        # Get how many chunks are needed for the image
        # as an array of integers
        image_chunks = np.ceil(image_size / chunk_size).astype(np.int64)

        chunk_dimensions = [chunk_size, chunk_size, number_of_bands]

        for x in range(image_chunks[0]):
            for y in range(image_chunks[1]):

                # print(f"X:{x}, Y:{y}")
                # print(chunk_dimensions)

                # Create an empty chunk with dimensions [chunk_size, chunk_size, number_of_bands]
                empty_chunk = np.zeros(chunk_dimensions)

                # Get the area of the hyperspectral image which should be captured by the chunk.
                # If chunk_size is not a factor of image_size, do not include pixels which are outside of the boundaries of the image. 
                chunk_start = np.array([x,y]) * chunk_size
                chunk_start = np.minimum(chunk_start, image_size)
                chunk_end = np.minimum(chunk_start + chunk_size, image_size)
                chunk_size = chunk_end - chunk_start

                # Get the chunk of the image.
                # The size of an image chunk can be smaller than the chunk_size iff
                # image_size is not a factor of chunk_size AND the chunk is at the right edge and/or bottom edge of the image.
                image_chunk = hyperspectral_image[chunk_start[0]:chunk_end[0], chunk_start[1]:chunk_end[1], :]

                # If the enforce_chunk_size flag is set to True,
                # Make all image chunks the same size as chunk_size by
                # padding any chunks that are smaller than chunk_size with
                # zeroes, retaining the original chunk at the top-left.
                if enforce_chunk_size:
                    
                    full_chunk = empty_chunk

                    # Fill the top-left of the empty chunk we created earlier with the contents of image_chunk
                    full_chunk[:chunk_size[0],:chunk_size[1],:] = image_chunk

                    # Now our chunk will be the same size as chunk_size
                    image_chunk = empty_chunk

                chunks.append(image_chunk)
        
        return chunks

    # Remove all chunks in a list of chunks where the percentage of background pixels are higher than a given percentage.
    # E.g., remove_background_chunks_by_percentage(chunks, 0.8) will remove all chunks with 20% or more background pixels.
    @staticmethod
    def remove_background_chunks_by_percentage(chunks, background_percentage):

        threshold = 1 - background_percentage

        filtered_chunks = filter((lambda x : Chunking.get_plant_ratio_in_chunk(x) <= threshold), chunks)
        return np.array(list(filtered_chunks))

class HSImage:
    
    # Load an ENVI Standard Hyperspectral Image into Memory.
    def load_envi_image(header_file, data_file):
        img = sp.envi.open(header_file, data_file)
        img = img.load()
        return img
    
    def save_envi_image(hdr, image, meta):
        # hdr = name of file to be used when saving image.
        # image = image data.
        # meta = additional metadata fields to be added to .hdr file as a dict.
        folder = 'chunks_{}/'.format(np.datetime64('today'))
        os.makedirs('./images/{}'.format(folder), exist_ok=True)
        sp.envi.save_image( hdr_file='./images/{}/{}.hdr'.format(folder, hdr),
                            image=image, 
                            dtype=np.float32,
                            interleave='bsq',
                            ext='raw',
                            metadata=meta,
                            force=True
                        )

    # Convert a loaded ENVI Standard Hyperspectral Image into a NumPy Array / Matrix.
    def to_matrix(img):
        img_matrix = np.array(img)
        return img_matrix

    # Load an ENVI Standard Hyperspectral Image as a matrix.
    @staticmethod
    def load_as_matrix(header_file, data_file):
        return HSImage.to_matrix(
            HSImage.load_envi_image(header_file, data_file)
            )

    def matrix_reshape_2d(matrix):
        # Reshape the image matrix to a 2 dimensional matrix. 
        # 2 spatial dimensions are merged into a single vector which can be factored out in later steps.
        matrix_reshape_2d = matrix.reshape(-1, matrix.shape[-1])
        return matrix_reshape_2d

    def calibrate_image(img_raw_matrix, dark_ref_matrix, white_ref_matrix):
        # Alpha: α(x, y, n) = Iraw(x, y, n) − RD (x, y, n)
        # Beta: β(x, y, n) = RW (x, y, n) − RD (x, y, n) − minx,y,n α(x, y, n)
        alpha = img_raw_matrix - dark_ref_matrix                                         
        beta = np.mean(white_ref_matrix, axis=(0,1)) - dark_ref_matrix - alpha.min()    
        calibrated_image = 100 * (alpha/beta)
        return calibrated_image
    
    def lloyd_kmeans_plus(matrix):
        kmeans = KMeans(
                    n_clusters=2, 
                    init='k-means++', 
                    n_init='auto', 
                    max_iter=50, 
                    tol=0.0001, 
                    verbose=0, 
                    random_state=None, 
                    copy_x=True, 
                    algorithm='lloyd'
                ).fit(matrix)
        return kmeans

    def segment_image(hsi_image, bands, red_wavelength_start = 710, red_wavelength_end = 740, vnir_wavelength_start = 650, vnir_wavelength_end = 680):
        return Segmentor.segment_image(hsi_image, bands, red_wavelength_start, red_wavelength_end, vnir_wavelength_start, vnir_wavelength_end)
    
        # Slice a hyperspectral image into an array of chunks.
        # Returns: NumPy array of chunks.
        
        # chunk_size:          Width and Height of chunks.

        # enforce_chunk_size:  Whether to force chunks to be of size chunk_size when the dimensions of the image are not a factor of chunk_size.
        #                      If true, chunks at the edges of the image will be padded with zeroes if they exceed the image size.
        
        # minimum_plant_ratio: Removes chunks where the percentage of background pixels is higher than a given amount.
        #                      E.g., 0.25: Remove all chunks with a ratio of at least 75% background pixels.       
    @staticmethod
    def slice_hsi_into_chunks(hsi_image, chunk_size, enforce_chunk_size = False, minimum_plant_ratio = 0):
        chunks = Chunking.slice_hsi_into_chunks(hsi_image, chunk_size=chunk_size, enforce_chunk_size=enforce_chunk_size)

        if minimum_plant_ratio > 0:
            chunks = Chunking.remove_background_chunks_by_percentage(chunks, minimum_plant_ratio)

        return chunks
    
    # Read a directory and return an array of all ENVI Standard HSI files.
    # Returns: Array of tuples (envi_header_absolute_file_path, envi_data_absolute_file_path)
    @staticmethod
    def get_envi_hsi_from_directory(directory, envi_header_extension = ".hdr", envi_data_extension = ".raw"):
        images = []

        files_in_directory = os.listdir(directory)

        # Filter out non-header files.
        header_files = list(filter((lambda x: x.endswith(envi_header_extension)), files_in_directory))

        # Get the raw files by stripping the header files of their extension and appending the data file extension.
        data_files = list(map((lambda x: x.rstrip(envi_header_extension) + envi_data_extension), header_files))

        # Make all relative file paths absolute.
        header_files = list(map((lambda x : os.path.join(directory,x)), header_files))
        data_files = list(map((lambda x : os.path.join(directory,x)), data_files))

        # Ensure that all HSI files exist in the directory.
        for header_file, data_file in zip(header_files, data_files):
            if not os.path.exists(header_file): raise FileNotFoundError(f"HSI Header File: {header_file} was not found.")
            if not os.path.exists(data_file): raise FileNotFoundError(f"Missing data file for HSI header file: {header_file}.\nExpected file {data_file} was not found.")
        
        return list(zip(header_files, data_files))

    # Pre-process a hyperspectral image

    # header_file:           The file path of the ENVI Standard HSI header file.
    # data_file:             The file path of the ENVI Standard HSI data file.
    # dark_reference_matrix: A matrix of the dark reference image.
    # save_path:             Where the resulting pre-processed file should be saved.
    # slice_into_chunks:     Whether to slice the hyperspectral image into separate chunks.
    #    chunk_size:                            The width and height of chunks to slice the HSI into.
    #    enforce_chunk_size:                    Whether to force chunks to be of size chunk_size when the dimensions of the image are not a factor of chunk_size.
    #                                           If true, chunks at the edges of the image will be padded with zeroes if they exceed the image size.
    #    exclude_chunks_with_background_ratio:  Removes chunks where the percentage of background pixels is higher than a given amount.
    #                                           E.g., 0.25: Remove all chunks with a ratio of at least 25% background pixels.   
    @staticmethod
    def preprocess_hsi_file(header_file, 
                            data_file,
                            dark_ref_hdr,
                            dark_ref_raw,
                            save_path,
                            label=None,
                            slice_into_chunks = True, chunk_size = 32, enforce_chunk_size = True, minimum_plant_ratio = 0.4):
        
        # Load envi format hyperspectral image
        image = HSImage.load_envi_image(header_file, data_file)
        # Load envi format dark reference image
        dark_ref_image = HSImage.load_envi_image(dark_ref_hdr, dark_ref_raw)
        # Extract bands from image
        bands = image.bands.centers

        # Convert images to numpy matrix for processing
        image_matrix = HSImage.to_matrix(image)
        dark_reference_matrix = HSImage.to_matrix(dark_ref_image)

        # Transform image_matrix to a 2 dimensional matrix [(axb), c]
        image_matrix_2d = HSImage.matrix_reshape_2d(image_matrix)

        # Image Calibration via K-Means clustering to extract white reference
        kmeans = HSImage.lloyd_kmeans_plus(image_matrix_2d)

        # Use result of kmeans to extract white reference image from image using cluster id=1
        white_ref_matrix = image_matrix_2d[kmeans.labels_ == 1]

        # Calibrate the image using the white reference and dark images (see calibrate_image method for more details)
        calibrated_image = HSImage.calibrate_image(image, dark_reference_matrix, white_ref_matrix)

        # Segment plant pixels from background pixels and pass image through denoising algorithm. 
        segmented_image = HSImage.segment_image(calibrated_image, bands)

        # Slice calibrated and segmented image into chunks of specified size.
        if slice_into_chunks:
            image_chunks = HSImage.slice_hsi_into_chunks(
                segmented_image,
                chunk_size = chunk_size,
                enforce_chunk_size = enforce_chunk_size,
                minimum_plant_ratio = minimum_plant_ratio
            )

        file_base = str(os.path.splitext(os.path.basename(header_file))[0])
        for i, chunk in enumerate(image_chunks):
            file_name = file_base+str(i)
            HSImage.save_envi_image(
                hdr=file_name,
                image=chunk,
                meta={
                    'label':label,
                    'file_name': file_name, 
                    'length':chunk.shape[0],
                    'width':chunk.shape[1],
                    'bands':chunk.shape[2],  
                }
            )

        #np.savez(save_path, image_chunks)
    
    @staticmethod
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    # Get the band ID of the band which is closest to a given wavelength.
    @staticmethod
    def get_closest_band_for_wavelength(bands, wavelength):
        return HSImage.find_nearest(bands, wavelength)


class CustomDataset(Dataset, HSImage):
    ## Ref: https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/2
    ## https://medium.com/analytics-vidhya/creating-a-custom-dataset-and-dataloader-in-pytorch-76f210a1df5d

    def __init__(self, image_folder, class_map):
        self.image_folder = image_folder
        self.file_list = glob.glob(self.image_folder+'\*.hdr')
        self.class_map = class_map
        self.data = []

        for file in self.file_list:
            base = os.path.splitext(file)[0]
            hdr = str(glob.glob(base+'.hdr')[0])
            raw = str(glob.glob(base+'.raw')[0])
            d = {}
            with open(hdr) as f:
                for line in f:
                    values = line.strip().split('=')
                    if len(values) > 1:
                        d[values[0].strip()] = values[1].strip()
            self.data.append([hdr, raw, d['label']])    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hdr_path, raw_path, label = self.data[index]
        image = HSImage.load_envi_image(hdr_path, raw_path)
        label = self.class_map[label]
        img_tensor = torch.from_numpy(image)
        img_tensor = img_tensor.permute(2, 0, 1)
        label = torch.tensor([label])
        return img_tensor, label