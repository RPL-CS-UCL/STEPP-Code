#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from os.path import join
import torch.nn.functional as F
import torch
import torch.quantization as quant
from torchvision import transforms as T
from omegaconf import OmegaConf
import numpy as np
from pytictac import Timer

from STEPP.DINO.backbone import get_backbone


class DinoInterface:
    def __init__(
        self,
        device: str,
        backbone: str = "dino",
        input_size: int = 448,
        backbone_type: str = "vit_small",
        patch_size: int = 8,
        projection_type: str = None,  # nonlinear or None
        dropout_p: float = 0,  # True or False
        pretrained_weights: str = None,
        interpolate: bool = True,
        use_mixed_precision: bool = False,
        cfg: OmegaConf = OmegaConf.create({}),
    ):
        # Load config
        if cfg.is_empty():
            self._cfg = OmegaConf.create(
                {
                    "backbone": backbone,
                    "backbone_type": backbone_type,
                    "input_size": input_size,
                    "patch_size": patch_size,
                    "projection_type": projection_type,
                    "dropout_p": dropout_p,
                    "pretrained_weights": pretrained_weights,
                    "interpolate": interpolate,
                }
            )
        else:
            self._cfg = cfg

        # Initialize DINO
        self._model = get_backbone(self._cfg)

        # Send to device
        self._model.to(device)
        self._device = device

        # self._model = quant.quantize_dynamic(self._model, dtype=torch.qint8, inplace=True)
        self.use_mixed_precision = use_mixed_precision
        if self.use_mixed_precision:
            self._model = self._model.to(torch.float16)


        # Other
        normalization = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self._transform = T.Compose(
            [
                T.Resize(input_size, T.InterpolationMode.NEAREST),
                T.CenterCrop(input_size),
                # T.CenterCrop((input_size, 1582)),
                normalization,
            ]
        )

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._model.to(device)
        self._device = device

    @torch.no_grad()
    def inference(self, img: torch.tensor):
        """Performance inference using DINO
        Args:
            img (torch.tensor, dtype=type.torch.float32, shape=(B,3,H.W)): Input image

        Returns:
            features (torch.tensor, dtype=torch.float32, shape=(B,D,H,W)): per-pixel D-dimensional features
        """

        # Resize image and normalize
        resized_img = self._transform(img).to(self._device)
        if self.use_mixed_precision:
            resized_img=resized_img.half()
        
        # Extract features
        features = self._model.forward(resized_img)
        # print('features shape before interpolation', features.shape)

        if self._cfg.interpolate:
            # resize and interpolate features
            B, D, H, W = img.shape
            new_features_size = (H, W)
            # pad = int((W - H) / 2)
            features = F.interpolate(features, new_features_size, mode="bilinear", align_corners=True)
            print('features shape after interpolation', features.shape)
            # features = F.pad(features, pad=[pad, pad, 0, 0])

        return features.to(torch.float32)

    @property
    def input_size(self):
        return self._cfg.input_size

    @property
    def backbone(self):
        return self._cfg.backbone

    @property
    def backbone_type(self):
        return self._cfg.backbone_type

    @property
    def vit_patch_size(self):
        return self._cfg.patch_size
    
    
def get_dino_features(img, dino_size, interpolate):
    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # #convert image to torch tensor
    # img = torch.from_numpy(img)
    img = img.to(device) 
    # img = F.interpolate(img, scale_factor=0.25)

    # Settings
    size = 896
    model = dino_size
    patch = 14
    backbone = "dinov2"

    # Inference with DINO
    # Create DINO
    di = DinoInterface(
        device=device,
        backbone=backbone,
        input_size=size,
        backbone_type=model,
        patch_size=patch,
        interpolate=interpolate,
    )

    # with Timer(f"DINO, input_size, {di.input_size}, model, {di.backbone_type}, patch_size, {di.vit_patch_size}"):
    feat_dino = di.inference(img)
    # print(f"Feature shape after interpolation: {feat_dino.shape}")

    return feat_dino

def average_dino_feature_segment(features, segment_img, segments=None):
    #features is a torch tensor of shape [1, 384, 64, 64]

    averaged_features = []

    if segments is None:
        segments = np.unique(segment_img)

    # Loop through each segment
    for segment_id in segments:
        segment_pixels = segment_img.astype(np.uint16) == segment_id
        selected_features = features[:, :, segment_pixels]
        vector = selected_features.mean(dim=-1)
        averaged_features.append(vector)

    # Stack all vectors vertically to form a m by n tensor
    averaged_features_tensor = torch.cat(averaged_features, dim=0)

    return averaged_features_tensor

def average_dino_feature_segment_tensor(features, segment_img, segments=None):

    if segments is None:
        segments, segments_count = torch.unique(segment_img, return_counts=True)

    features_flattened = features.permute(0,2,3,1).flatten(0,-2) # (bhw x n_features)
    index = segment_img.flatten().unsqueeze(-1).repeat(1,features_flattened.shape[-1]).long() # (bhw x n_features)
    num_segments = torch.max(segment_img).int()+1 # adding +1 for the 0 ID.
    output = torch.zeros( (num_segments, features_flattened.shape[-1]), device="cuda", dtype=features.dtype)
    segment_means = output.scatter_reduce(0,index, features_flattened, reduce="sum")
    segment_means = segment_means[segment_means.sum(-1)!=0] / segments_count.unsqueeze(-1)
    # print("Difference between two methods",(segment_means-averaged_features_tensor).sum())
    averaged_features_tensor = segment_means

    return averaged_features_tensor

def run_dino_interfacer():
    """Performance inference using DINOv2VIT and stores result as an image."""

    from pytictac import Timer
    from seb_trav.utils.misc import get_img_from_fig, load_test_image, make_results_folder, remove_axes
    import matplotlib.pyplot as plt

    #supress warnings
    import warnings
    warnings.filterwarnings("ignore")


    # Create test directory
    outpath = make_results_folder("test_dino_interfacer")

    # Inference model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_test_image().to(device)
    # img = F.interpolate(img, scale_factor=0.25)

    print('image after interpolation before going to model', img.shape)

    plot = False
    save_features = True

    # Settings
    size = 896
    model = "vit_small"
    patch = 14
    backbone = "dinov2"

    # Inference with DINO
    # Create DINO
    di = DinoInterface(
        device=device,
        backbone=backbone,
        input_size=size,
        backbone_type=model,
        patch_size=patch,
    )

    with Timer(f"DINO, input_size, {di.input_size}, model, {di.backbone_type}, patch_size, {di.vit_patch_size}"):
        feat_dino = di.inference(img)
    print(f"Feature shape after interpolation: {feat_dino.shape}")

    if save_features:
        for i in range(5):
            fig = plt.figure(frameon=False)
            fig.set_size_inches(2, 2)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(feat_dino[0][i].cpu(), cmap=plt.colormaps.get("inferno"))

            # Store results to test directory
            out_img = get_img_from_fig(fig)
            out_img.save(
                join(
                    outpath,
                    f"forest_clean_dino_feat{i:02}_{di.input_size}_{di.backbone_type}_{di.vit_patch_size}.png",
                )
            )
            plt.close("all")

    if plot:
        # Plot result as in colab
        fig, ax = plt.subplots(10, 11, figsize=(1 * 11, 1 * 11))

        for i in range(10):
            for j in range(11):
                if i == 0 and j == 0:
                    continue

                elif (i == 0 and j != 0) or (i != 0 and j == 0):
                    ax[i][j].imshow(img.permute(0, 2, 3, 1)[0].cpu())
                    ax[i][j].set_title("Image")
                else:
                    n = (i - 1) * 10 + (j - 1)
                    if n >= di.get_feature_dim():
                        break
                    ax[i][j].imshow(feat_dino[0][n].cpu(), cmap=plt.colormaps.get("inferno"))
                    ax[i][j].set_title("Features [0]")
        remove_axes(ax)
        plt.tight_layout()

        # Store results to test directory
        out_img = get_img_from_fig(fig)
        out_img.save(
            join(
                outpath,
                f"forest_clean_{di.backbone}_{di.input_size}_{di.backbone_type}_{di.vit_patch_size}.png",
            )
        )
        plt.close("all")


if __name__ == "__main__":
    run_dino_interfacer()
