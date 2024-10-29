import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
import os
from pytictac import Timer
import warnings
import argparse

from seb_trav import ROOT_DIR
from seb_trav.DINO import run_dino_interfacer
from seb_trav.DINO.dino_feature_extract import DinoInterface
from seb_trav.SLIC.slic_segmentation import SLIC
from seb_trav.utils import misc
from seb_trav.utils.misc import load_image
from seb_trav.DINO.dino_feature_extract import get_dino_features, average_dino_feature_segment


class FeatureDataSet:
    def __init__(self, path_to_image_folder, path_to_pixels):
        self.img_width = 1408#1280
        self.img_height = 1408#720
        self.x_boarder = 0 #20
        self.y_boarder = 0 #30
        self.start_image_idx = 0#750
        self.interpolate = False
        self.dino_size = 'vit_small'
        self.use_mixed_precision = True

        if self.dino_size == 'vit_small':
            self.feature_dim = 384
        elif self.dino_size == 'vit_base':
            self.feature_dim = 768
        elif self.dino_size == 'vit_large':
            self.feature_dim = 1024
        elif self.dino_size == 'vit_giant':
            self.feature_dim = 1536

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Settings
        self.size = 700
        self.dino_size = "vit_small"
        self.patch = 14
        self.backbone = "dinov2"

        # Inference with DINO
        # Create DINO
        self.di = DinoInterface(
                device=self.device,
                backbone=self.backbone,
                input_size=self.size,
                backbone_type=self.dino_size,
                patch_size=self.patch,
                interpolate=False,
                use_mixed_precision=self.use_mixed_precision,
            )

        #points
        with open(path_to_pixels, 'r') as f:
            path_pixels = json.load(f)

        path_pixels_resized = []
        #remove entries that contain values of larger than 720-20 and 1280-30
        for pixels in path_pixels:
            pixels = [pixel for pixel in pixels if pixel[0] < (self.img_width-self.y_boarder) and pixel[1] < (self.img_height-self.x_boarder)]
            #also take off 20 from the x and 30 from the y
            pixels = [(pixel[0] - self.x_boarder, pixel[1] - self.y_boarder) for pixel in pixels]

            path_pixels_resized.append(pixels)
        self.path_pixels_resized = path_pixels_resized

        print("loaded pixels")
        
        #images
        self.images = sorted([os.path.join(path_to_image_folder, img) for img in os.listdir(path_to_image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
        #what does this do?
        if len(self.images) > len(self.path_pixels_resized):
            self.images = self.images[:-(len(self.images) -len(self.path_pixels_resized))]

        print("loaded images")

    
def main(feat):

    #supress warnings
    warnings.filterwarnings("ignore")
    slic = SLIC(crop_x=0, crop_y=0)
    average_features_segments = np.zeros((1, feat.feature_dim))

    for i in range(len(feat.images)):
        if feat.path_pixels_resized[i] == []:
            continue
        img = cv2.imread(feat.images[i])
        segments, segmented_image = slic.Slic_segmentation_for_given_pixels(feat.path_pixels_resized[i], img)
        resized_segmented_image, new_segment_dict = slic.make_masks_smaller_numpy(segments.keys(), segmented_image, int(feat.size/feat.patch))

        tensor_img = load_image(feat.images[i]).to(feat.device)

        #get dino features
        features = feat.di.inference(tensor_img)

        #average dino features over segments
        average_features = average_dino_feature_segment(features, resized_segmented_image, new_segment_dict.keys())
        #convert to numpy array
        average_features = average_features.cpu().detach().numpy()
        average_features_segments = np.concatenate((average_features_segments,average_features), axis=0)

        print('processed image:', i,'/', len(feat.images))#, end='\r')
    
    average_features_segments = average_features_segments[1:]

    print('\n')
    print('average_features_segments shape:', average_features_segments.shape)

    return average_features_segments


if __name__ == '__main__':

    #add arguments if its being called from the command line
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--number', type=str, help='path to the right richmond image folder')
    args = parser.parse_args()
    
    number = args.number
    filepath = '/home/sebastian/ARIA/aria_recordings/Richmond_forest/'

    path_to_image_folder = filepath + '/mps_Richmond_forest_'+number+'_vrs/rgb'
    path_to_pixels = filepath + '/mps_Richmond_forest_'+number+'_vrs/ARIA_richmond_forest_'+number+'_pixels.json'
    data_preprocessing = FeatureDataSet(path_to_image_folder, path_to_pixels, )
    dataset = main(data_preprocessing)

    #save dataset
    result_folder = misc.make_results_folder('Richmond_forest_dataset_ump')
    np.save(result_folder + '/richmond_forest_DINO_ump_vit_small_size_700_number_'+number, dataset)