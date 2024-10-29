import torch
from seb_trav.model.mlp import ReconstructMLP
from seb_trav.utils.misc import load_image
from seb_trav.SLIC.slic_segmentation import SLIC
from seb_trav.utils.make_dataset import FeatureDataSet
from seb_trav.DINO.dino_feature_extract import DinoInterface, get_dino_features, average_dino_feature_segment
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch.nn as nn
from matplotlib.colors import LinearSegmentedColormap
import time
import torch.nn.functional as F
import warnings
from PIL import Image as PILImage
import seaborn as sns
from pytictac import Timer

warnings.filterwarnings("ignore")


def test_feature_reconstructor(mode, model_path, image_path, thresh):
    # mode = 1 for running segmentwise inference
    # mode = 2 for running whole image inference

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # load the model
    model = ReconstructMLP(384,[256, 128, 64, 32, 64, 128, 256]) # [256, 32, 384])  #
    #load the model with the weights
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    return test_feature_reconstructor_with_model(mode, model, image_path, thresh)

def test_feature_reconstructor_with_model(mode,model, image_path, thresh):
    start = time.time()

    alpha = 0.5

    #load an image
    img = cv2.imread(image_path)
    torch_img = load_image(image_path)
    H, W, D = img.shape
    H1 = 64
    new_features_size = (H, H)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(img, (new_features_size))
    # small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = thresh#= 0.1

    # Settings
    size = 700
    dino_size = "vit_small"
    patch = 14
    backbone = "dinov2"

    # Inference with DINO
    # Create DINO
    di = DinoInterface(
            device=device,
            backbone=backbone,
            input_size=size,
            backbone_type=dino_size,
            patch_size=patch,
            interpolate=False,
            use_mixed_precision = False,
        )

    torch_img = torch.from_numpy(small_image)
    torch_img = torch_img.permute(2, 0, 1)  
    torch_img = (torch_img.type(torch.float32) / 255)[None].to(device)
    # torch_img.to(self.device)
    dino_size = 'vit_small'
    # features = get_dino_features(torch_img, dino_size, False)
    features = di.inference(torch_img)

    print('features shape',features.shape)

    if mode == 'segment_wise':
        #segment the whole image and get each pixel for each segment value
        slic = SLIC(crop_x=0, crop_y=0)
        segments, segmented_image = slic.Slic_segmentation_for_all_pixels(small_image)
        print('segmented image shape:', segmented_image.shape)
        resized_segmented_img, new_segment_dict = slic.make_masks_smaller_numpy(segments, segmented_image, 50)

        #average the features over the segments
        average_features = average_dino_feature_segment(features, resized_segmented_img)
        
        # Forward pass the entire batch
        reconstructed_features = model(average_features)

        # Calculate the losses for the entire batch
        loss_fn = nn.MSELoss(reduction='none')
        losses = loss_fn(average_features, reconstructed_features)
        losses = losses.mean(dim=1).cpu().detach().numpy()  # Average the losses across the feature dimension
    
        #set the segment values of the segmented image to equal the loss in losses
        for key, loss in zip(new_segment_dict.keys(), losses):
            segmented_image = np.where(segmented_image == int(key), loss, segmented_image)

        segmented_image - np.where(segmented_image > 10, 10, segmented_image)

        # Normalize the segmented image values to the range [0, 0.15]
        segmented_image = (segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min()) * 0.45

        # Change all values above 1 to 1
        segmented_image = np.where(segmented_image > threshold, threshold, segmented_image)
        # segmented_image = np.where(segmented_image < self.threshold, 0.0, segmented_image)

        # # Calculate the extent to center the segmented image
        # original_height, original_width = small_image.shape[:2]
        # segmented_height, segmented_width = segmented_image.shape[:2]

        # # Crop the original image to the segmented image size
        # x_offset = (original_width - segmented_width) // 2
        # y_offset = (original_height - segmented_height) // 2
        # small_image = img[y_offset:y_offset + segmented_height, x_offset:x_offset + segmented_width]

        # Create the colormap
        s = 0.3  # If bigger, get more fine-grained green, if smaller get more fine-grained red
        cmap = cm.get_cmap("RdYlBu", 256)  # or RdYlGn
        cmap = np.vstack([
            cmap(np.linspace(0, s, 128)), 
            cmap(np.linspace(1 - s, 1.0, 128))
        ])  # Stretch the colormap
        cmap = (cmap[:, :3] * 255).astype(np.uint8)

        # Reverse the colormap if needed
        cmap = cmap[::-1]

        # Normalize the segmented image values to the range [0, 255]
        segmented_normalized = ((segmented_image - segmented_image.min()) / 
                                (segmented_image.max() - segmented_image.min()) * 255).astype(np.uint8)

        # Map the segmented image values to colors
        color_mapped_img = cmap[segmented_normalized]

        # Convert images to RGBA
        img_rgba = PILImage.fromarray(np.uint8(small_image)).convert("RGBA")
        seg_rgba = PILImage.fromarray(color_mapped_img).convert("RGBA")

        # Adjust the alpha channel to vary the transparency
        seg_rgba_np = np.array(seg_rgba)
        alpha_channel = seg_rgba_np[:, :, 3]  # Extract alpha channel
        alpha_channel = (alpha_channel * 1.0).astype(np.uint8)  # Adjust transparency (50% transparent)
        seg_rgba_np[:, :, 3] = alpha_channel  # Update alpha channel
        seg_rgba = PILImage.fromarray(seg_rgba_np)

        # Alpha composite the images
        img_new = PILImage.alpha_composite(img_rgba, seg_rgba)
        img_rgb = img_new.convert("RGB")

        #resize the image to the original size
        img_rgb = img_rgb.resize((W,H))

        # Overlay the segmented image on the original image
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title(mode + '_reconstruction_' + dino_size + '_threshold_' + str(threshold))
        plt.axis('off')

    elif mode == 'pixel_wise':

        # torch shape is (1, 384, 64, 64)
        features = features.permute(2, 3, 1, 0)

        #change the shape to (4096, 384)
        features_tensor = features.reshape(50*50, 384)

        with Timer('Inference: '):
            # Forward pass the entire batch
            reconstructed_features = model(features_tensor)

        # Calculate the losses for the entire batch
        loss_fn = nn.MSELoss(reduction='none')
        losses = loss_fn(features_tensor, reconstructed_features)
        losses = losses.mean(dim=1).cpu().detach().numpy()  # Average the losses across the feature dimension

        #reshape losses to be 64x64
        losses = losses.reshape(50, 50)

        #resize the cost map to the original image size
        cost_map = cv2.resize(losses, (H, H))
       
        print('time to run inference:', time.time()-start)

        cost_map = np.where(cost_map > 10,10, cost_map)


        # Normalize the segmented image values to the range [0, 0.15]
        cost_map = (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min()) * 0.45


        #change all values above 1 to 1
        # cost_map = np.where(cost_map < 3, 0, cost_map)
        cost_map = np.where(cost_map > threshold, threshold, cost_map)

        # Create the colormap
        s = 0.3  # If bigger, get more fine-grained green, if smaller get more fine-grained red
        cmap = cm.get_cmap("RdYlBu", 256)  # or RdYlGn
        cmap = np.vstack([
            cmap(np.linspace(0, s, 128)), 
            cmap(np.linspace(1 - s, 1.0, 128))
        ])  # Stretch the colormap
        cmap = (cmap[:, :3] * 255).astype(np.uint8)

        # Reverse the colormap if needed
        cmap = cmap[::-1]

        # Normalize the segmented image values to the range [0, 255]
        cost_map_normalized = ((cost_map - cost_map.min()) / 
                                (cost_map.max() - cost_map.min()) * 255).astype(np.uint8)

        # Map the segmented image values to colors
        color_mapped_img = cmap[cost_map_normalized]

        # Convert images to RGBA
        img_rgba = PILImage.fromarray(np.uint8(small_image)).convert("RGBA")
        seg_rgba = PILImage.fromarray(color_mapped_img).convert("RGBA")

        # Adjust the alpha channel to vary the transparency
        seg_rgba_np = np.array(seg_rgba)
        alpha_channel = seg_rgba_np[:, :, 3]  # Extract alpha channel
        alpha_channel = (alpha_channel * 0.75).astype(np.uint8)  # Adjust transparency (50% transparent)
        seg_rgba_np[:, :, 3] = alpha_channel  # Update alpha channel
        seg_rgba = PILImage.fromarray(seg_rgba_np)

        # Alpha composite the images
        img_new = PILImage.alpha_composite(img_rgba, seg_rgba)
        img_rgb = img_new.convert("RGB")

        #resize the image to the original size
        img_rgb = img_rgb.resize((W,H))

        # Overlay the segmented image on the original image
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title(mode + '_reconstruction_' + dino_size + '_threshold_' + str(threshold))
        plt.axis('off')

    # plt.show()
    return fig
    
if __name__ == '__main__':
    model_path = '/home/sebastian/Documents/code/seb_trav/results/trained_model/richmond_forest_full_ViT_small_big_nn_checkpoint_20240821-1825.pth'
    # model_path = '/home/sebastian/Documents/code/seb_trav/results/trained_model/OPS_lab_02_ViT_small_big_nn_checkpoint_20240808-2235.pth'
    image_path = '/home/sebastian/Documents/small_forest.png'
    threshold = 0.15
    test_feature_reconstructor('segment_wise',model_path, image_path, threshold)

    #save figure to test folder
    count = time.strftime("%Y%m%d-%H%M")
    plt.savefig('/home/sebastian/Documents/code/seb_trav/tests/test'+ count +'.png')
    plt.show()