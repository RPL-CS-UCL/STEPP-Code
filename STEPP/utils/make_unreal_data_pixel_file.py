import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
from pytictac import Timer

def overlay_images(n1_path, n2_path):
        n1_image = cv2.imread(n1_path)
        n2_image = cv2.imread(n2_path)
        # n2_image[..., 3] = 1
    
        mask = n2_image != 0
    
        # Create an output image with all black pixels
        output_image = np.zeros_like(n1_image)
    
        # Apply the mask to n1_image and store the result in output_image
        output_image[mask] = n1_image[mask]

        output_image[0:520] = 0

        #create a list of pixel coord pairs where the image is not black
        pixels = []
        non_black_pixels = np.argwhere(np.any(output_image != 0, axis=-1))
        pixels = non_black_pixels[:, ::-1].tolist()

        return output_image, pixels

path_to_image_folder = '/home/sebastian/Documents/Unreal_data/SebDatasets/CityParkDataset/1'
path_to_trajectory_folder = '/home/sebastian/Documents/Unreal_data/SebDatasets/CityParkDataset/2'

images = sorted([os.path.join(path_to_image_folder, img) for img in os.listdir(path_to_image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
trajectory_images = sorted([os.path.join(path_to_trajectory_folder, img) for img in os.listdir(path_to_trajectory_folder) if img.endswith((".png", ".jpg", ".jpeg"))])



all_pixels = []
for i in range(len(images)):
    output_img, pixels = overlay_images(images[i], trajectory_images[i])
    
    all_pixels.append(pixels)

    print('processed image:', i,'/', len(images), end='\r')

#save the pixels to json
path = '/home/sebastian/Documents/Unreal_data/SebDatasets/CityParkDataset/CityPark_pixels.json'
with open(path, 'w') as f:
    json.dump(all_pixels, f)

print('Finished saving pixels to:\n', path)
