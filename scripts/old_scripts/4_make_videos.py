import os
from shapeAnalysis import utils as shp_utils
import argparse
from shapeAnalysis.video_utils import save_segmentation_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    assert os.path.exists(DATA_DIR), "DATA_DIR not found: {}".format(DATA_DIR)
    assert os.path.exists(PROJECT_DIR), "PROJECT_DIR not found: {}".format(PROJECT_DIR)

    segmentations_subdir = "segmentations"

    segm_dir = os.path.join(PROJECT_DIR, segmentations_subdir)
    paths_input_images = shp_utils.get_paths_input_files(segm_dir, ".h5")
    raw_data_dir = DATA_DIR

    for full_path, rel_path in paths_input_images:
        if "_postProcSegm.h5" in full_path:
            # Read raw data and segmentation:
            final_segmentation = shp_utils.readHDF5(full_path, "data")
            raw_path = os.path.join(raw_data_dir, rel_path).replace("_postProcSegm.h5", ".h5")
            raw = shp_utils.readHDF5(raw_path, "data")

            # Create video:
            save_segmentation_video(raw, final_segmentation,
                                    full_path.replace("_postProcSegm.h5", ".mp4"),
                                    nb_frames=200)
            print(rel_path, ": movie created")


