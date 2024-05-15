from farmers import CustomDataset as cd

FOLDER_PATH = r"C:\Users\manse\Documents\gitrepos\HyperspectralPlantImaging\images\chunks_2024-05-15"
class_map = {'mosaic':0, 'smut':1, 'healthy':2}
dataset = cd(image_folder=FOLDER_PATH, class_map=class_map)
