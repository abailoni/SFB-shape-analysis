from PIL import Image
import skimage.io
import os
import imageio
from shapeAnalysis import utils as shp_utils
import numpy as np


# TODO: update
from pathutils.get_dir_paths import get_scratch_dir
root_data_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/bckgr_subtraction")
out_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/segmentations")

paths_input_images = shp_utils.get_paths_input_files(root_data_dir, ".h5")

path_ilastik = "/home/abailoni_local/ilastik-1.3.3post3-Linux/run_ilastik.sh"
path_ilastik_proj = os.path.join(get_scratch_dir(), "projects/SFB_viktor/pix_classifc-v1.ilp")

for full_path, rel_path in paths_input_images:
    # Compose ilastik command:
    file_out_dir = os.path.join(out_dir, rel_path)
    file_out_dir = os.path.split(file_out_dir)[0]
    shp_utils.check_dir_and_create(file_out_dir)
    command = "{} --headless --readonly --input_axes=tyx --project={} --export_source=\"Simple Segmentation\" --output_filename_format={}/{{nickname}}_results.hdf5  --output_format=\"compressed hdf5\" --export_dtype=uint8 {}".format(path_ilastik, path_ilastik_proj, file_out_dir, full_path)
    # print(command)
    stream = os.popen(command)
    output = stream.read()

# shp_utils.check_dir_and_create(root_out_data_dir)

print("Done")

