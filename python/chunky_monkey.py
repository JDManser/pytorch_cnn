from farmers import HSImage as hsi
import glob
import os

IMG_PATH = 'D:\\SRA_Sugarcane_Data(Original HSI)\\test\\'
DARK_REF_RAW = "D:\SRA_Sugarcane_Data(Original HSI)\dark_ref_irradiance.raw"
DARK_REF_HDR = "D:\SRA_Sugarcane_Data(Original HSI)\dark_ref_irradiance.hdr"
files = glob.glob(IMG_PATH+"*.raw")

for file in files:
    base = os.path.splitext(os.path.basename(file))[0]
    hdr = str(glob.glob(IMG_PATH+base+'.hdr')[0])
    raw = str(glob.glob(IMG_PATH+base+'.raw')[0])

    mosaic = ['Q44', 'CP29-116', 'Q68', 'Q78', 'Q82']
    smut = ['Nco310', 'Q124', 'Q171', 'Q205', 'Q208']
    healthy = ['_c']

    if any(n in base for n in healthy):
        target = 'healthy'
    elif any(n in base for n in smut):
        target = 'smut'
    elif any(n in base for n in mosaic):
        target = 'mosaic'

    hsi.preprocess_hsi_file(
        hdr, 
        raw, 
        DARK_REF_HDR, 
        DARK_REF_RAW,  
        minimum_plant_ratio=0.4,
        chunk_size=64,
        label=target
    )