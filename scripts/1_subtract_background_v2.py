from PIL import Image
import skimage.io
import os
import imageio
from shapeAnalysis import utils as shp_utils
import numpy as np


# TODO: update
from pathutils.get_dir_paths import get_scratch_dir
root_data_dir = os.path.join(get_scratch_dir(), "datasets/SFB_viktor/raw_images")
root_out_data_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data")

paths_input_images = shp_utils.get_paths_input_files(root_data_dir, ".h5")

for full_path, rel_path in paths_input_images:
    video = shp_utils.readHDF5(full_path, "data")
    video = video.astype('float32') / 255.
    print(full_path, video.shape)
    median_values = np.median(video, axis=0)
    normalized_video = video - median_values

    # Re-normalize so that median value is fixed at value 128:
    mask1 = normalized_video < 0.
    mask2 = np.logical_not(mask1)
    normalized_video_1 = normalized_video / normalized_video.min() # Array with values in [0, 1]
    normalized_video_2 = normalized_video / normalized_video.max()# Array with values in [0, 127]
    normalized_video[mask1] = normalized_video_1[mask1] * (-128.) # Array with values in [-128, 0]
    normalized_video[mask2] = normalized_video_2[mask2] * 127. # Array with values in [0, 127]
    normalized_video += 128.
    normalized_video = normalized_video.astype('uint8')

    # Write results:
    out_subdir = "bckgr_subtraction_v2"
    out_file = os.path.join(root_out_data_dir, out_subdir, rel_path)
    # Create out dir:
    out_dir = os.path.split(out_file)[0]
    shp_utils.check_dir_and_create(out_dir)

    shp_utils.writeHDF5(normalized_video, out_file, "data")

print("Done")

