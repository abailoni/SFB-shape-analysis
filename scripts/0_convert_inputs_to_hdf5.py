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

paths_input_images = shp_utils.get_paths_input_files(root_data_dir, ".tif")

for full_path, rel_path in paths_input_images:
    # Read video:
    video = skimage.io.imread(full_path)
    # Some files are kind of broken, so we need to read them in different way:
    if video.ndim != 3:
        video = imageio.mimread(full_path)
        video = np.array(video)

    video = video.astype('uint8')


    # Write results:
    out_file = full_path.replace(".tif", ".h5")
    shp_utils.writeHDF5(video, out_file, "data")



