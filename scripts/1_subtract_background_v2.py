import os
from shapeAnalysis import utils as shp_utils
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    assert os.path.exists(DATA_DIR), "DATA_DIR not found: {}".format(DATA_DIR)
    preproc_subdir = "preprocessed"

    # Get list of all .h5 images in the DATA_DIR folder:
    paths_input_images = shp_utils.get_paths_input_files(DATA_DIR, ".h5")

    for full_path, rel_path in paths_input_images:
        # Read data:
        video = shp_utils.readHDF5(full_path, "data")

        # Normalize betwen 0 and 1 and take median along time-dimension:
        video = video.astype('float32') / 255.
        print(full_path, video.shape)
        median_values = np.median(video, axis=0)
        normalized_video = video - median_values

        # Re-normalize so that median value is fixed at value 128:
        mask1 = normalized_video < 0.
        mask2 = np.logical_not(mask1)
        normalized_video_1 = normalized_video / normalized_video.min() # Array with values in [0, 1]
        normalized_video_2 = normalized_video / normalized_video.max()# Array with values in [0, 1]
        normalized_video[mask1] = normalized_video_1[mask1] * (-128.) # Array with values in [-128, 0]
        normalized_video[mask2] = normalized_video_2[mask2] * 127. # Array with values in [0, 127]
        normalized_video += 128.
        normalized_video = normalized_video.astype('uint8')

        # Write results:
        out_file = os.path.join(PROJECT_DIR, preproc_subdir, rel_path)
        out_dir = os.path.split(out_file)[0]
        shp_utils.check_dir_and_create(out_dir)
        shp_utils.writeHDF5(normalized_video, out_file, "data")


