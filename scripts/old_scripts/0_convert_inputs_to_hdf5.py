import skimage.io
import os
import imageio
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

    # Get list of all .tif images in the DATA_DIR folder:
    paths_input_images = shp_utils.get_paths_input_files(DATA_DIR, ".tif")

    for full_path, rel_path in paths_input_images:
        # Load video:
        video = skimage.io.imread(full_path)
        # Some files are kind of broken, so we need to read them in different way:
        if video.ndim != 3:
            video = imageio.mimread(full_path)
            video = np.array(video)

        video = video.astype('uint8')

        # Write video in hdf5:
        out_file = full_path.replace(".tif", ".h5")
        shp_utils.writeHDF5(video, out_file, "data")
        print(rel_path, "written")


