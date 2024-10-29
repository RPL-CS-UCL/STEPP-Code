from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import cv2
from seb_trav import ROOT_DIR

def make_results_folder(name):
    path = os.path.join(ROOT_DIR, "results", name)
    os.makedirs(path, exist_ok=True)
    return path

def get_img_from_fig(fig, dpi=180):
    """Returns an image as numpy array from figure

    Args:
        fig (matplotlib.figure.Figure): Input figure.
        dpi (int, optional): Resolution. Defaults to 180.

    Returns:
        buf (np.array, dtype=np.uint8 or PIL.Image.Image): Resulting image.
    """
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    buf = np.asarray(buf)
    buf = Image.fromarray(buf)
    buf = buf.convert("RGB")
    return buf

def load_test_image():
    np_img = cv2.imread(os.path.join(ROOT_DIR, "/home/sebastian/Documents/anymal_experiment_rosbag/000280.png"))
    np_img = np_img[200:-200, 200:-200]
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    return img

def load_image(path):
    np_img = cv2.imread(path)
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]
    return img

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])

def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def save_dataset(dataset, path):
    #create a folder if it does not exist
    folder = os.path.dirname(path + '/' + 'dataset')
    os.makedirs(folder, exist_ok=True)
    np.save(folder, dataset)
