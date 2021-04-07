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
    video_without_bkg = np.abs(video - median_values)

    # Re-normalize between 0 and 255:
    video_without_bkg = video_without_bkg - video_without_bkg.min()
    video_without_bkg = ((video_without_bkg / video_without_bkg.max()) * 255.).astype('uint8')


    # Write results:
    out_subdir = "bckgr_subtraction"
    out_file = os.path.join(root_out_data_dir, out_subdir, rel_path)
    # Create out dir:
    out_dir = os.path.split(out_file)[0]
    shp_utils.check_dir_and_create(out_dir)

    shp_utils.writeHDF5(video_without_bkg, out_file, "data")

print("Done")

