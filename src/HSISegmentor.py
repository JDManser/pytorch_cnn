

from spectral import *
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import *

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
    def segment_image(hsi_image, bands = None, red_wavelength_start = 710, red_wavelength_end = 740, vnir_wavelength_start = 650, vnir_wavelength_end = 680):
        if not bands:
            bands = hsi_image.bands.centers
        
        # Get the NDVI of the image
        ndvi = Segmentor.__get_ndvi_using_ranges(hsi_image, bands, red_wavelength_start, red_wavelength_end, vnir_wavelength_start, vnir_wavelength_end)

        threshold = threshold_otsu(ndvi)

        binary_segment = np.squeeze(ndvi > threshold)

        eroded_image = Segmentor.__denoise_segmentation_mask(binary_segment)

        image_segment_mask = eroded_image

        segmented_image = Segmentor.__apply_segmentation_mask(hsi_image, image_segment_mask)

        return segmented_image
        