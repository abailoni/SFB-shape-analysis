import os
from shapeAnalysis import utils as shp_utils
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    parser.add_argument('--ilastik_path', type=str) # For example: "/home/user/ilastik-1.3.3post3-Linux/run_ilastik.sh"
    parser.add_argument('--ilastik_project_path', type=str) # For example: "/home/user/pix_classifc-v3.ilp"
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    path_ilastik = args.ilastik_path
    path_ilastik_proj = args.ilastik_project_path
    assert os.path.exists(PROJECT_DIR), "PROJECT_DIR not found: {}".format(PROJECT_DIR)

    preproc_subdir = "preprocessed"
    segmentations_subdir = "segmentations"

    paths_input_images = shp_utils.get_paths_input_files(os.path.join(PROJECT_DIR, preproc_subdir), ".h5")
    out_dir = os.path.join(PROJECT_DIR, segmentations_subdir)

    for full_path, rel_path in paths_input_images:
        # Create out-dir:
        file_out_dir = os.path.join(out_dir, rel_path)
        file_out_dir = os.path.split(file_out_dir)[0]
        shp_utils.check_dir_and_create(file_out_dir)

        # Run ilastik in headless mode:
        command = "{} --headless --readonly --input_axes=tyx --project={} --export_source=\"Simple Segmentation\" --output_filename_format={}/{{nickname}}_results.hdf5  --output_format=\"compressed hdf5\" --export_dtype=uint8 {}".format(path_ilastik, path_ilastik_proj, file_out_dir, full_path)
        stream = os.popen(command)
        output = stream.read()
        print(rel_path, "processed.")


