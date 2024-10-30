#file to run SLIC segmentation on an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
from fast_slic import Slic
import time
import json
from collections import defaultdict
from torchvision import transforms as T
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from pytictac import Timer

class SLIC():
    def __init__(self, crop_x=30, crop_y=20, num_superpixels=400, compactness=15):
        if crop_x == 0 and crop_y == 0:
            self.crop = False
        else:
            self.crop = True
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slic = Slic(num_components=self.num_superpixels, compactness=self.compactness)

    def Slic_segmentation_for_given_pixels(self, pixels, image):
        # Load your image
        if self.crop:
            only_img = image[self.crop_y:-self.crop_y, self.crop_x:-self.crop_x]
        else:
            only_img = image
        # Convert BGR image to RGB for matplotlib
        image_rgb = cv2.cvtColor(only_img, cv2.COLOR_BGR2RGB)

        # Create Slic object
        slic = Slic(num_components=self.num_superpixels, compactness=self.compactness)

        # Perform segmentation
        segmented_image = slic.iterate(image_rgb)
        
        # Assuming pixels is a list of (x, y) tuples or a 2D array where each row is an (x, y) pair
        pixels_array = np.array(pixels)

        # Extract the x and y coordinates
        y_coords = pixels_array[:, 0]
        x_coords = pixels_array[:, 1]

        # Use advanced indexing to get the segment values at the given (x, y) coordinates
        segment_values = segmented_image[x_coords, y_coords]

        # Create a dictionary to hold lists of pixel coordinates for each segment
        segment_dict = defaultdict(list)

        # Populate the dictionary with pixel coordinates grouped by their segment
        for i in range(len(segment_values)):
            segment = segment_values[i]
            pixel = (x_coords[i], y_coords[i])
            segment_dict[segment].append(pixel)

        return segment_dict, segmented_image
    
    def Slic_segmentation_for_all_pixels(self, image):
        # Load your image
        if self.crop:
            only_img = image[self.crop_y:-self.crop_y, self.crop_x:-self.crop_x]
        else:
            only_img = image

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(only_img, cv2.COLOR_BGR2RGB)

        # Create Slic object
        slic = Slic(num_components=self.num_superpixels, compactness=self.compactness)

        # Perform segmentation
        segmented_image = self.slic.iterate(image_rgb)

        # Get unique segment values
        unique_segments = np.unique(segmented_image)

        return unique_segments, segmented_image
    
    def Slic_segmentation_for_all_pixels_torch(self, image):
        # Load your image
        if self.crop:
            only_img = image[self.crop_y:-self.crop_y, self.crop_x:-self.crop_x]
        else:
            only_img = image

        # Convert BGR image to RGB
        image_rgb = cv2.cvtColor(only_img, cv2.COLOR_BGR2RGB)

        # Create Slic object
        slic = Slic(num_components=self.num_superpixels, compactness=self.compactness)

        # Perform segmentation
        segmented_image = self.slic.iterate(image_rgb)


        #put image onto the gpu
        segmented_image = torch.from_numpy(segmented_image).to(self.device)

        # Get unique segment values
        unique_segments = torch.unique(segmented_image)

        return unique_segments, segmented_image
    
    def make_masks_smaller_numpy(self, segment_values, segmented_image, wanted_size):
        # Convert NumPy array to PIL image
        segmented_image_pil = Image.fromarray(segmented_image.astype('uint16'), mode='I;16')


        # Resize the image while maintaining the pixel values
        resized_segmented_image_pil = segmented_image_pil.resize((wanted_size, wanted_size), Image.NEAREST)
        
        # Convert the resized PIL image back to a NumPy array
        resized_segmented_image = np.array(resized_segmented_image_pil).astype(np.uint16)

        new_segment_dict = defaultdict(list)

        # Iterate over each unique segment value
        for key in segment_values:
            # Find the coordinates where the pixel value equals the key
            coordinates = np.where(resized_segmented_image == key)
            
            # Zip the coordinates to get (row, column) pairs and store them in the dictionary
            new_segment_dict[key].extend(zip(coordinates[0], coordinates[1]))

        return resized_segmented_image, new_segment_dict
    
    def make_masks_smaller_torch(self, segment_values, segmented_image, wanted_size, return_dict=True):
        
        segmented_image = segmented_image.unsqueeze(0).unsqueeze(0).float()
        # Resize the image while maintaining the pixel values
        resized_segmented_image = F.interpolate(
            segmented_image,
            size=(wanted_size, wanted_size),
            mode='nearest')

        #get rid of the first and second dimension
        resized_segmented_image = resized_segmented_image.squeeze(0).squeeze(0)

        new_segment_dict = defaultdict(list)
        if return_dict:
            # Iterate over each unique segment value
            with Timer("loop"):
                for key in segment_values:
                    # Find the coordinates where the pixel value equals the key
                    coordinates = torch.where(resized_segmented_image == key)
                    
                    # Zip the coordinates to get (row, column) pairs and store them in the dictionary
                    new_segment_dict[key].extend(zip(coordinates[0].tolist(), coordinates[1].tolist()))
            print (f"looped {len(segment_values)} times")
        
        return resized_segmented_image, new_segment_dict
    
    def get_difference_pixels(img1, img2):
        # Compute the absolute difference
        difference = cv2.absdiff(img1, img2)

        # Threshold the difference to find the significant changes
        _, thresholded_difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)

        # Convert to grayscale
        gray_diff = cv2.cvtColor(thresholded_difference, cv2.COLOR_BGR2GRAY)

        # Find contours in the thresholded difference
        contours, _ = cv2.findContours(gray_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        flattened_list = [item[0].tolist() for item in largest_contour]
        
        return flattened_list
    
def run_SLIC_segmentation():
    """Run SLIC on an image and visualize the segmented image"""

    ##############################################
    # This should all be coming from a config file
    ##############################################
    img_width = 1408
    img_height = 1408
    x_boarder = 200
    y_boarder = 200
    number = 10
    # pixels = path[number]
    # img_path = images[number]
    img_path = 'path_to_test_image'
    print('img_path:', img_path)
    # ##############################################

    # #plot the image with the pixels
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #crop image to remove the boarder
    img = img[y_boarder:-y_boarder, x_boarder:-x_boarder]
    plt.figure(figsize=(10, 10))
    plt.imshow(img)#, cmap='inferno')
    plt.axis('off')
    # plt.show()

    # def overlay_images_1(n1_path, n2_path):
    #     n1_image = cv2.imread(n1_path)
    #     n2_image = cv2.imread(n2_path)
    #     # n2_image[..., 3] = 1
    
    #     mask = n2_image != 0
    
    #     # Create an output image with all black pixels
    #     output_image = np.zeros_like(n1_image)
    
    #     # Apply the mask to n1_image and store the result in output_image
    #     output_image[mask] = n1_image[mask]

    #     output_image[0:520] = 0

    #     #create a list of pixel coord pairs where the image is not black
    #     pixels = []
    #     non_black_pixels = np.argwhere(np.any(output_image != 0, axis=-1))
    #     pixels = non_black_pixels[:, ::-1].tolist()

    #     return output_image, pixels
    
    # def overlay_images_2(n1_path, n2_path):
    #     n1_image = cv2.imread(n1_path)
    #     n2_image = cv2.imread(n2_path)
    #     # n2_image[..., 3] = 1
    
    #     mask = n2_image != 0
    
    #     # Create an output image with all black pixels
    #     output_image = np.zeros_like(n1_image)
    
    #     # Apply the mask to n1_image and store the result in output_image
    #     output_image[mask] = n1_image[mask]

    #     output_image[0:520] = 0

    #     #create a list of pixel coord pairs where the image is not black
    #     pixels = []
    #     for i in range(output_image.shape[0]):
    #         for j in range(output_image.shape[1]):
    #             if np.any(output_image[i, j] != 0):
    #                 pixels.append([j, i])

    #     return output_image, pixels
    

    # with Timer('overlay_images'):
    #     output_img_1, pixels_1 = overlay_images_1(img_path, 'path_to_test_image')

    # with Timer('overlay_images'):
    #     output_img_2, pixels_2 = overlay_images_2(img_path, 'path_to_test_image')

    # print(pixels_1 == pixels_2)
    # print(pixels_1[:10])

    # exit()

    # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(output_img)#, cmap='inferno')
    # plt.axis('off')

    #remove entries that contain values of larger than 720-20 and 1280-30
    # pixels = [pixel for pixel in pixels if pixel[0] < (img_width-y_boarder) and pixel[1] < (img_height-x_boarder)]
    # #also take off 20 from the x and 30 from the y
    # pixels = [(pixel[0] - x_boarder, pixel[1] - y_boarder) for pixel in pixels]

    slic = SLIC(crop_x=0, crop_y=0, num_superpixels=100, compactness=10)
    seg, seg_img = slic.Slic_segmentation_for_all_pixels(img)
    # segments, segmented_image = slic.Slic_segmentation_for_given_pixels(pixels, img)


    print('number of unique values in segmented image:', len(np.unique(seg_img)))
    print(seg)
    segmented_image_mask = seg_img

    #make values in each segment in seg_img random:
    # for value in seg:
    #     random_value = np.random.randint(0, 255)  # Generate a single random value for the current segment
    #     seg_img = np.where(seg_img == value, random_value, seg_img)    # seg_img = np.random.randint(0, len(np.unique(seg_img)), (seg_img.shape[0], seg_img.shape[1]))

    unique_values = set()  # To keep track of the unique random values assigned
    for value in seg:
        random_value = np.random.randint(0, 255)
        
        # Ensure the random_value hasn't already been used
        while random_value in unique_values:
            random_value = np.random.randint(0, 255)  # Generate a new random value if a collision occurs
        
        # Assign the unique random value and record it
        seg_img = np.where(seg_img == value, random_value, seg_img)
        unique_values.add(random_value)  # Add to set of used values
    print(len(unique_values))
    pixel_list = [[420, 973],
                  [484, 833],
                  [475, 745],
                  [550, 778],
                  [520, 717],
                  [585, 678],
                  [683, 632],
                  [610, 610],
                  [660, 668],
                  [475,1000]]
    values = []
    for pixels in pixel_list:
        # point = (pixel[1], pixel[0])
        val = seg_img[(pixels[1], pixels[0])]
        print('val:', val)
        values.append(val)

    segmented_image_mask = np.where(np.isin(seg_img, values), seg_img, 0)
    segmented_image_mask_expanded = np.expand_dims(segmented_image_mask, axis=-1)  # Adds a third dimension

    # Now segmented_image_mask_expanded will have shape (1008, 1008, 1)
    # Use np.where to compare and select values
    seg_img_path = np.where(segmented_image_mask_expanded != 0, img, 255)
    # for pixel in pixels:
    #     # point = (pixel[1], pixel[0])
    #     val = segmented_image[(pixel[1], pixel[0])]
    #     segmented_image_mask = np.where(segmented_image == val, 0, segmented_image_mask)

    # Optionally, visualize the segmented image
    plt.figure(figsize=(10, 10))
    plt.imshow(seg_img)#, cmap='inferno')
    plt.axis('off')

    plt.figure(figsize=(10, 10))
    plt.imshow(segmented_image_mask)#, cmap='inferno')
    plt.axis('off')

    plt.figure(figsize=(10, 10))
    plt.imshow(seg_img_path)#, cmap='inferno')
    plt.axis('off')

    # resized_segmented_image, new_segment_dict = slic.make_masks_smaller(segments.keys(), segmented_image, 64)

    # print('new_segment_dict:', new_segment_dict)

    # print('number of unique values in resized segmented image:', len(np.unique(resized_segmented_image)))

    # resized_segmented_image_mask = resized_segmented_image

    # for key in new_segment_dict.keys():
    #     resized_segmented_image_mask = np.where(resized_segmented_image == float(key), 0, resized_segmented_image_mask)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(resized_segmented_image)#, cmap='inferno')
    # plt.axis('off')

    # # Optionally, visualize the resized segmented image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(resized_segmented_image_mask)#, cmap='inferno')
    # plt.axis('off')
    plt.show()


if __name__ == "__main__":
    run_SLIC_segmentation()