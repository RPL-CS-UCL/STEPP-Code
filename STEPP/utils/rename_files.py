import numpy
import cv2
import os

def rename_files_in_folder(folder_path):
    for i, filename in enumerate(os.listdir(folder_path)):
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, f"{int(filename[:-4]):06d}.png"))

if __name__ == '__main__':
    folder_path = 'path_to_folder'
    rename_files_in_folder(folder_path)