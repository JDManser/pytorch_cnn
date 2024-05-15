from farmers import HyperspectralImage as hsi
# Fully Automated Recognition by Machine learning Employing Remote Sensing

def analyse_hyperspectral_image(hsi_header_file, hsi_data_file, dark_reference_header_file, dark_reference_data_file):
    image = hsi.load_image(hsi_header_file, hsi_data_file)
    
    return NotImplementedError()