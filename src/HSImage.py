import spectral as sp
import numpy as np
from HSISegmentor import Segmentor
from HSIChunking import Chunking
from sklearn.cluster import KMeans
from spectral import *
import numpy as np
import os
import pandas as pd

class HSImage:
    
    # Load an ENVI Standard Hyperspectral Image into Memory.
    def load_image(header_file, data_file):
        img = sp.envi.open(header_file, data_file)
        img = img.load()
        return img

    # Convert a loaded ENVI Standard Hyperspectral Image into a NumPy Array / Matrix.
    def to_matrix(img):
        img_matrix = np.array(img)
        return img_matrix

    # Load an ENVI Standard Hyperspectral Image as a matrix.
    @staticmethod
    def load_as_matrix(self, header_file, data_file):
        return self.to_matrix(
            self.load_image(header_file, data_file)
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

    def segment_image(hsi_image, bands = None, red_wavelength_start = 710, red_wavelength_end = 740, vnir_wavelength_start = 650, vnir_wavelength_end = 680):
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
    def get_envi_hsi_images_in_directory(directory, envi_header_extension = ".hdr", envi_data_extension = ".raw"):
        images = []

        files_in_directory = os.listdir(directory)

        # Filter out non-header files.
        header_files = list(filter((lambda x: x.endswith(envi_header_extension)), files_in_directory))

        # Get the raw files by stripping the header files of their extension and appending the data file extension.
        data_files = list(map((lambda x: x.rstrip(envi_header_extension) + envi_data_extension), header_files))

        # Make all relative file paths absolute.
        header_files = list(map((lambda x : os.path.join(directory,x)), header_files))
        data_files = list(map((lambda x : os.path.join(directory,x)), data_files))

        files = []

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
    def preprocess_hsi_file(
                header_file, 
                data_file,
                dark_ref_hdr,
                dark_ref_raw,
                save_path,
                target=None,
                base=None,
                slice_into_chunks = True, chunk_size = 32, enforce_chunk_size = True, minimum_plant_ratio = 0
        ):
        
        image = HSImage.load_image(header_file, data_file)
        image = image.load()
        dark_ref_image = HSImage.load_image(dark_ref_hdr, dark_ref_raw)
        bands = image.bands.centers

        image_matrix = HSImage.to_matrix(image)
        image_matrix_2d = HSImage.matrix_reshape_2d(image_matrix)

        dark_reference_matrix = HSImage.to_matrix(dark_ref_image)

        # Image Calibration via K-Means clustering to extract white reference
        kmeans = HSImage.lloyd_kmeans_plus(image_matrix_2d)
        white_ref_matrix = image_matrix_2d[kmeans.labels_ == 1]
        calibrated_image = HSImage.calibrate_image(image, dark_reference_matrix, white_ref_matrix)

        # Image segmentation
        segmented_image = HSImage.segment_image(calibrated_image, bands)

        if slice_into_chunks:
            image_chunks = HSImage.slice_hsi_into_chunks(
                segmented_image,
                chunk_size = chunk_size,
                enforce_chunk_size = enforce_chunk_size,
                minimum_plant_ratio = minimum_plant_ratio
            )
        '''
        if os.path.exists(save_path+'annotations.csv'):
            df = pd.read_csv(save_path+'annotations.csv')
        else:
            df = pd.DataFrame(columns=['file', 'length', 'width', 'bands', 'target'])
            df.set_index('file')
        
        for i, chunk in enumerate(image_chunks):
            np.savez(save_path+'train/'+base+str(i), chunk)
            row = {
                'file':base+str(i), 
                'length':chunk.shape[0],
                'width':chunk.shape[1],
                'bands':chunk.shape[2],
                'target':target
            }
            df.loc[len(df)] = row
        df.to_csv(save_path+'annotations.csv', index=None)
        '''
        for i, chunk in enumerate(image_chunks):
            sp.envi.save_image(str(save_path+'train/'+base+str(i)+'.hdr'),
                               chunk, 
                               dtype=np.float32,
                               interleave='bsq',
                               ext='raw'
                            )
            #np.savez(save_path+'train/'+base+str(i), chunk)

