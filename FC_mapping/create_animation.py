from PIL import Image
import os
import cv2
import numpy as np
import re



# Input/output paths
input_folder = "CH_fraction_monthly"      # folder containing your PNGs
output_video = "CH_FC_grassland.mp4"

# Cropping box (left, upper, right, lower)
crop_box = (50, 50, 450, 450)  # change as needed

# Output video settings
fps = 1  # frames per second

# Get sorted list of image files
image_files = [
    f for f in os.listdir(input_folder)
    if f.endswith(".png") and 'global_grass_lu_ndsi2' in f
]
image_files.sort(key=lambda f: int(re.search(r'CH_fraction_(\d+)_', f).group(1)))

# Use the first image to determine size
first_image = Image.open(os.path.join(input_folder, image_files[0]))
width, height = first_image.size
cropped_height = height // 3

# OpenCV expects (width, height) in reverse order
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, cropped_height))

for filename in image_files:
    path = os.path.join(input_folder, filename)
    with Image.open(path) as img:
        width, height = img.size
        cropped = img.crop((0, 0, width, height // 3))
        frame = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

video_writer.release()
print("MP4 video created:", output_video)