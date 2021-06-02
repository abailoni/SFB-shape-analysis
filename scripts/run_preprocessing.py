import os
from shapeAnalysis import utils as shp_utils
from shapeAnalysis import functions as shp_functions
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    assert os.path.exists(DATA_DIR), "DATA_DIR not found: {}".format(DATA_DIR)

    # Get names of the all the vides to be processed:
    video_names = []
    for (dirpath, dirnames, filenames) in os.walk(DATA_DIR):
        video_names = dirnames
        break

    # Create project directory:
    shp_utils.check_dir_and_create(PROJECT_DIR)

    # Run pipeline:
    print("\n\n\n\n----------------------- \nImporting tif files...")
    shp_functions.convert_tif_images_to_hdf5_stack(DATA_DIR, PROJECT_DIR, video_names)
    print("\n\n\n\n----------------------- \nPreprocessing videos...")
    shp_functions.pre_process_all_videos(PROJECT_DIR, video_names)






