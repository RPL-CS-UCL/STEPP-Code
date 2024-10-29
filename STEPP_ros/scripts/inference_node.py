#!/Rocket_ssd/miniconda3/envs/STEPP/bin/python3

import rospy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import torch
import cv2
from cv_bridge import CvBridge
import numpy as np
import torch.nn as nn
import time
from PIL import Image as PILImage
from torchvision import transforms
# import seaborn as sns
from matplotlib import cm
import warnings
from queue import Queue
from threading import Thread, Lock

from STEPP.DINO.backbone import get_backbone
from STEPP.DINO.dino_feature_extract import DinoInterface
from STEPP.DINO.dino_feature_extract import get_dino_features, average_dino_feature_segment, average_dino_feature_segment_tensor
from STEPP.SLIC.slic_segmentation import SLIC
from STEPP.model.mlp import ReconstructMLP
from STEPP_ros.msg import Float32Stamped

warnings.filterwarnings("ignore")
CV_BRIDGE = CvBridge()
TO_TENSOR = transforms.ToTensor()
TO_PIL_IMAGE = transforms.ToPILImage()

from pytictac import Timer

class InferenceNode:
    def __init__(self):
        self.image_queue = Queue(maxsize=1)
        self.lock = Lock()

        self.processing = False
        self.image_sub = rospy.Subscriber('/camera/color/image_raw/compressed', CompressedImage, self.image_callback)
        self.inference_pub = rospy.Publisher('/inference/result', Float32MultiArray, queue_size=200)
        self.inference_stamped_pub = rospy.Publisher('/inference/results_stamped_post', Float32Stamped, queue_size=200)
        self.visu_traversability_pub = rospy.Publisher('/inference/visu_traversability_post', Image, queue_size=200)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Threshold for traversability
        self.threshold = 0.2

        # Settings
        self.size = 700
        self.dino_size = "vit_small"
        self.patch = 14
        self.backbone = "dinov2"
        self.ump = rospy.get_param('~ump', True)
        self.cutoff = rospy.get_param('~cutoff', 1.2)
        print(self.cutoff)
        print(type(self.cutoff))

        # Inference with DINO
        # Create DINO
        self.di = DinoInterface(
                device=self.device,
                backbone=self.backbone,
                input_size=self.size,
                backbone_type=self.dino_size,
                patch_size=self.patch,
                interpolate=False,
                use_mixed_precision = self.ump,
            )

        self.slic = SLIC(crop_x=0, crop_y=0)

        # Load model architecture 
        # self.model = ReconstructMLP(384, [256, 64, 32, 16, 32, 64, 256])
        self.model = ReconstructMLP(384, [256, 128, 64, 32, 64, 128, 256])

        # Load model weights
        state_dict = torch.load(rospy.get_param('~model_path'))
        self.model.load_state_dict(state_dict)

        # Move model to the device
        self.model.to(self.device)

        self.visualize = rospy.get_param('~visualize', False)

        self.thread = Thread(target=self.process_images)
        self.thread.start()

        print('Inference node initialized')

    def publish_matrix(self, matrix):
        msg = Float32MultiArray()
        msg.data = matrix.flatten().tolist()  # Flatten the matrix and convert to list
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "rows"
        msg.layout.dim[0].size = matrix.shape[0]
        msg.layout.dim[0].stride = matrix.shape[1]  # stride is the number of columns
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[1].label = "columns"
        msg.layout.dim[1].size = matrix.shape[1]
        msg.layout.dim[1].stride = 1  # stride is 1 for columns
        self.inference_pub.publish(msg)

    def publish_array_stamped(self, matrix):
        msg = Float32Stamped()

        # Get the current time in nanoseconds
        msg.header.stamp = rospy.Time.now()

        msg.data = Float32MultiArray()
        msg.data.data = matrix.flatten().tolist()  # Flatten the matrix and convert to list
        msg.data.layout.dim.append(MultiArrayDimension())
        msg.data.layout.dim[0].label = "rows"
        msg.data.layout.dim[0].size = matrix.shape[0]
        msg.data.layout.dim[0].stride = matrix.shape[1]  # stride is the number of columns
        msg.data.layout.dim.append(MultiArrayDimension())
        msg.data.layout.dim[1].label = "columns"
        msg.data.layout.dim[1].size = matrix.shape[1]
        msg.data.layout.dim[1].stride = 1  # stride is 1 for columns
        self.inference_stamped_pub.publish(msg)

    def process_images(self):
        while not rospy.is_shutdown():
        # with Timer("Full loop"):
            image_data = self.image_queue.get()
            if image_data is None:
                break

            with self.lock:
                if isinstance(image_data, CompressedImage):
                    cv_image = CV_BRIDGE.compressed_imgmsg_to_cv2(image_data, desired_encoding="bgr8")
                else:
                    cv_image = CV_BRIDGE.imgmsg_to_cv2(image_data, desired_encoding="bgr8")

                try:
                    traversability_array, inference_img = self.inference_image(cv_image)
                except Exception as e:
                    print(f'Error: {e}')
                    self.processing = False
                    continue

                # self.publish_matrix(traversability_array)
                self.publish_array_stamped(traversability_array)

                if self.visualize:
                    self.visu_traversability_pub.publish(CV_BRIDGE.cv2_to_imgmsg(np.array(inference_img), "rgb8"))

            self.processing = False
            # print('-'*10)

    def inference_image(self, image):
        # Load an image
        org_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(org_img, (self.size, self.size))
        H, W, D = org_img.shape

    # with Timer("DINO feature extraction"):
        # Get the dino features
        torch_img = torch.from_numpy(img)
        torch_img = torch_img.permute(2, 0, 1)  
        torch_img = (torch_img.type(torch.float32) / 255)[None].to(self.device)
        # torch_img.to(self.device)
        dino_size = 'vit_small'
        # features = get_dino_features(torch_img, dino_size, False)
        features = self.di.inference(torch_img)

        # Segment the whole image and get each pixel for each segment value
    # with Timer("SLIC"):
        segments, segmented_image = self.slic.Slic_segmentation_for_all_pixels_torch(img)
    # with Timer("Make masks smaller"):
        resized_segmented_img, new_segment_dict = self.slic.make_masks_smaller_torch(segments, segmented_image, int(self.size/self.patch), return_dict=False)
    # Average the features over the segments
    # with Timer("Average dino feature"):
        average_features = average_dino_feature_segment_tensor(features, resized_segmented_img).to(self.device)
    # with Timer("Forward pass"):
        # Forward pass the entire batch
        reconstructed_features = self.model(average_features)

        # Calculate the losses for the entire batch
    # with Timer("Loss calculation"):
        loss_fn = nn.MSELoss(reduction='none')
        losses = loss_fn(average_features, reconstructed_features)
        losses = losses.mean(dim=1).cpu().detach().numpy()  # Average the losses across the feature dimension

    # with Timer("Set segment values optimized"):
        segmented_image = segmented_image.cpu().detach().numpy()
        # Get the unique keys from the resized segmented image
        unique_keys = np.unique(resized_segmented_img.cpu().detach().numpy()).astype(int)
        # Create an array that maps the unique segment values to the corresponding losses
        max_segment_value = np.max(segmented_image)
        default_loss = 1.0
        mapping_array = np.full(max_segment_value + 1, default_loss)
        # Fill the mapping array with the corresponding losses
        mapping_array[unique_keys] = losses
        # Use the mapping array to replace values in segmented_image
        segmented_image = mapping_array[segmented_image]

        #cuttoff the values at 10
        segmented_image = np.where(segmented_image > 10, 10, segmented_image)

        # Normalize the segmented image values to the range [0, 0.15]
        segmented_image = ((segmented_image - segmented_image.min()) / (segmented_image.max() - segmented_image.min())) * self.cutoff

        # Change all values above 1 to 1
        segmented_image = np.where(segmented_image > self.threshold, self.threshold, segmented_image)
        # segmented_image = np.where(segmented_image < self.threshold, 0.0, segmented_image)

        if self.visualize:
        # with Timer("image processing"):
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
            img_rgba = PILImage.fromarray(np.uint8(img)).convert("RGBA")
            seg_rgba = PILImage.fromarray(color_mapped_img).convert("RGBA")

            # Adjust the alpha channel to vary the transparency
            seg_rgba_np = np.array(seg_rgba)
            alpha_channel = seg_rgba_np[:, :, 3]  # Extract alpha channel
            alpha_channel = (alpha_channel * 0.5).astype(np.uint8)  # Adjust transparency (50% transparent)
            seg_rgba_np[:, :, 3] = alpha_channel  # Update alpha channel
            seg_rgba = PILImage.fromarray(seg_rgba_np)

            # Alpha composite the images
            img_new = PILImage.alpha_composite(img_rgba, seg_rgba)
            img_rgb = img_new.convert("RGB")

            #resize the image and the segmented image to the original size
            img_rgb = img_rgb.resize((W,H))
            segmented_image = cv2.resize(segmented_image, (W,H))

            return segmented_image, img_rgb
        else:
            segmented_image = cv2.resize(segmented_image, (W,H))
            
            return segmented_image, None

    def image_callback(self, data):
        if not self.processing:
            with self.lock:
                if not self.image_queue.full():
                    self.image_queue.put(data)
                    self.processing = True

if __name__ == '__main__':
    print('Starting inference node')
    rospy.init_node('inference_node')
    node = InferenceNode()
    rospy.spin()
