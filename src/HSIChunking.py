import numpy as np
#from spectral import *
#from PIL import Image
#import spectral.io.envi as envi

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