#file to run SLIC segmentation on an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fast_slic import Slic
import time
import json

def SLIC_segmentation(path):
    # Load your image
    image_path = path
    image = cv2.imread(image_path)
    img_2 = cv2.imread('/home/sebastian/Documents/ANYmal_data/OPS_grass/Masked_data/odom_data_masked/masks/mask_1318.png', cv2.IMREAD_GRAYSCALE)
    overlay_img = cv2.imread('/home/sebastian/Documents/ANYmal_data/OPS_grass/Masked_data/odom_data_masked/mask_overlay/trajectory_1318.png')
    only_img = image[20:-20, 30:-30]
    only_img_2 = img_2[20:-20, 30:-30]
    # Convert BGR image to RGB for matplotlib
    image_rgb = cv2.cvtColor(only_img, cv2.COLOR_BGR2RGB)

    print('shape of image:', image_rgb.shape)

    #read json file
    with open('all_points.json', 'r') as f:
        path_points = json.load(f)
    

    #create numpy array of the image
    img_2 = np.array(only_img_2)
    #make array binary
    img_2 = np.where(img_2 > 0, 1, 0)

    # Parameters
    num_superpixels = 500  # Adjust the number of superpixels
    compactness = 10.0     # Adjust the compactness factor

    # Create Slic object
    slic = Slic(num_components=num_superpixels, compactness=compactness)

    # Perform segmentation
    start2 = time.time()
    segmented_image = slic.iterate(image_rgb)
    end2 = time.time()
    print('Time to segment image:', end2-start2)

    print('type of segmented image:', type(segmented_image))
    print('shape of segmented image:', segmented_image.shape)
    print('Number of segments:', len(np.unique(segmented_image)))
    print(len(segmented_image[0]))

    # Optionally, visualize the segmentation
    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image, cmap='inferno')
    plt.axis('off')
    plt.show()

    #also plot an image with the segments overlayed on the original image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_2)
    # plt.imshow(segmented_image, alpha=0.85, cmap='inferno')
    # plt.axis('off')
    # plt.show()

    # #plot the biggest segment
    # #find the biggest segment
    # print(segmented_image.shape)
    # unique, counts = np.unique(segmented_image, return_counts=True)
    # print(unique)
    # print(counts)
    # biggest_segment = unique[np.argmax(counts)]
    # print(biggest_segment)
    # biggest_segment_mask = np.where(segmented_image == biggest_segment, 1, 0)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(biggest_segment_mask, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # plot only segments that are overlapping with the img_2 mask
    # segmented_mask = segmented_image * img_2
    # np.unique(segmented_mask)
    # print(len(np.unique(segmented_mask)))
    # for patch in np.unique(segmented_image):
    #     if patch not in np.unique(segmented_mask):
    #         segmented_image = np.where(segmented_image == patch, 0, segmented_image)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(segmented_image, cmap='inferno')
    # plt.imshow(image_rgb, alpha=0.5, cmap='inferno')
    # plt.axis('off')
    # plt.title('SLIC segmentation with mask')
    # plt.show()

    # #plot only segments that have path_points in them
    # # print(path_points[2213])
    # point_value = []

    # for i in range(len(path_points[2213])):
    #     point_value.append(segmented_image[path_points[2213][i][1], path_points[2213][i][0]])

    # for patch in np.unique(segmented_image):   
    #     if patch not in point_value:
    #         segmented_image = np.where(segmented_image == patch, 0, segmented_image)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(segmented_image, cmap='inferno')
    # plt.imshow(image_rgb, alpha=0.5, cmap='inferno')
    # plt.axis('off')
    # plt.title('SLIC segmentation with path points')
    # plt.show()


if __name__ == "__main__":
    SLIC_segmentation('/home/sebastian/Documents/ANYmal_data/OPS_grass/odom_chosen_images_2/004285.png')
