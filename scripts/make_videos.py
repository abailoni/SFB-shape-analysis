from PIL import Image
import skimage.io
import os
import imageio
from shapeAnalysis import utils as shp_utils
import numpy as np

from shapeAnalysis.video_utils import save_segmentation_video


# TODO: update
from pathutils.get_dir_paths import get_scratch_dir
raw_data_dir = os.path.join(get_scratch_dir(), "datasets/SFB_viktor/raw_images")
root_data_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/segmentations_v3")
out_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/segmentations_v3")

paths_input_images = shp_utils.get_paths_input_files(root_data_dir, ".h5")

for full_path, rel_path in paths_input_images:
    if "_postProcSegm.h5" in full_path:
        # Read raw data and segmentation:
        final_segmentation = shp_utils.readHDF5(full_path, "data")
        raw_path = os.path.join(raw_data_dir, rel_path).replace("_postProcSegm.h5", ".h5")
        raw = shp_utils.readHDF5(raw_path, "data")

        save_segmentation_video(raw, final_segmentation, full_path.replace("_postProcSegm.h5", ".mp4"), nb_frames=200)
        print(rel_path, "Done")
        # break

# shp_utils.check_dir_and_create(root_out_data_dir)

print("Done")

