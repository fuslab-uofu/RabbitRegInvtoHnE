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


# --- Config: edit these per run ---
RABBIT = 'R23-295'
BLOCK = 7

directory = f'/System/Volumes/Data/ceph/hifu/animal_data/IACUC1800/{RABBIT}/{RABBIT}_BlockFaceImages/block{BLOCK:02d}'
files = sorted(load_files_pathlib(directory, 'scatter', '.jpg'))

new_dir_path = Path(f"/System/Volumes/Data/ceph/hifu/users/jbonaventura/RabbitRegistrationProj/RabbitData/{RABBIT}/BlockFace_RGB/Block{BLOCK:02d}/CroppedImages")

# Create the directory and any necessary parents, suppressing errors if it exists
new_dir_path.mkdir(parents=True, exist_ok=True)
print(f"Directory '{new_dir_path}' created or already exists.")
print(f"Found {len(files)} files in {directory}")

# --- Preview crop on the center file, confirm or adjust before batch-cropping ---
crop_coords = [775, 3300, 1500, 3400]  # y1, y2, x1, x2

center_file = files[len(files) // 2]
while True:
    y1, y2, x1, x2 = crop_coords
    preview_crop = np.array(Image.open(center_file))[y1:y2, x1:x2, :]

    plt.imshow(preview_crop)
    plt.title(f"{center_file.name}  crop=({y1},{y2},{x1},{x2})")
    plt.show()

    answer = input("Does this crop look good? (y/n): ").strip().lower()
    if answer == 'y':
        break
    new_coords = input("Enter new crop coords as y1,y2,x1,x2: ")
    crop_coords = [int(v.strip()) for v in new_coords.split(',')]

y1, y2, x1, x2 = crop_coords
print(f"Using crop region: [{y1}:{y2}, {x1}:{x2}]")

# --- Batch crop + save ---
for file_path in files:
    print(file_path)
    # Open the image file
    img = Image.open(file_path)

    # Convert the Image object to a NumPy array
    img_array = np.array(img)
    cropped_im = img_array[y1:y2, x1:x2, :]

    # Save the image as a TIFF file
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    new_file_path = os.path.join(new_dir_path, file_name + ".tiff")
    CImage = Image.fromarray(cropped_im)
    CImage.save(new_file_path, 'TIFF')
