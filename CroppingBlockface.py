from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

def load_files_pathlib(directory_path, name_pattern, file_type):
    p = Path(directory_path)
    # The pattern combines the name pattern and file type
    pattern = f"*{name_pattern}*{file_type.strip('*')}"

    # Use rglob() for recursive search (subdirectories included) or glob() for only the current directory
    matching_files = list(p.rglob(pattern))  # Use p.glob(pattern) for non-recursive search

    return matching_files


directory = '/Users/jbonaventura/Downloads/R23-292/BlockFaceRGB/block07'
files = load_files_pathlib(directory, 'scatter', '.jpg')

# Define the path
new_dir_path = Path(os.path.join(directory, "CroppedImages"))

# Create the directory and any necessary parents, suppressing errors if it exists
new_dir_path.mkdir(parents=True, exist_ok=True)
print(f"Directory '{new_dir_path}' created or already exists.")

print(f"Found {len(files)} files:")
for file_path in files:
    print(file_path)
    # Open the image file
    img = Image.open(file_path)

    # Convert the Image object to a NumPy array
    img_array = np.array(img)
    #Change Crop region-
    cropped_im=img_array[775:3300,1550:3400,:]
    # plt.imshow(cropped_im)
    # plt.show()

    # # Save the image as a TIFF file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    new_file_path=os.path.join(new_dir_path, file_name + ".tiff")
    CImage = Image.fromarray(cropped_im)
    CImage.save(new_file_path, 'TIFF')