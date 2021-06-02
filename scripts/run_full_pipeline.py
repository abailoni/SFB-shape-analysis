import os
from shapeAnalysis import utils as shp_utils
from shapeAnalysis import functions as shp_functions
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    parser.add_argument('--ilastik_path', type=str) # For example: "/home/user/ilastik-1.3.3post3-Linux/run_ilastik.sh"
    parser.add_argument('--ilastik_project_path', type=str) # For example: "/home/user/pix_classifc-v3.ilp"
    parser.add_argument('--size_thresh', type=int, default=40)
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    path_ilastik = args.ilastik_path
    path_ilastik_proj = args.ilastik_project_path
    size_thresh = args.size_thresh
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
    print("\n\n\n\n----------------------- \nRunning ilastik...")
    shp_functions.run_ilastik_on_all_videos(PROJECT_DIR, video_names, path_ilastik, path_ilastik_proj)
    print("\n\n\n\n----------------------- \nPost-processing videos...")
    shp_functions.post_process_videos(PROJECT_DIR, video_names, size_thresh)






