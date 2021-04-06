from PIL import Image
import skimage.io
import os
import imageio
from shapeAnalysis import utils as shp_utils
import vigra
import numpy as np

from nifty import tools as ntools
import tifffile

SIZE_THRESH = 40

def get_map_of_segment_sizes_in_segmentation(segmentation):
    """
    Compute a size map of each segment in the given segmentation
    """
    node_sizes = np.bincount(segmentation.flatten())
    return ntools.mapFeaturesToLabelArray(segmentation, node_sizes[:, None], nb_threads=-1).squeeze()


# TODO: update
from pathutils.get_dir_paths import get_scratch_dir
root_data_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/segmentations_v3")
out_dir = os.path.join(get_scratch_dir(), "projects/SFB_viktor/processed_data/segmentations_v3")

paths_input_images = shp_utils.get_paths_input_files(root_data_dir, ".h5")

for full_path, rel_path in paths_input_images:
    if "_results.h5" in full_path:
        # Load segmentation:
        # Label 1 --> foreground
        # Label 2 --> background
        final_segmentation = shp_utils.readHDF5(full_path, "exported_data")
        # Set background to zero, convert and get rid of last dimension:
        final_segmentation = final_segmentation.astype('uint32')[..., 0]
        final_segmentation[final_segmentation==2] = 0

        # Run connected components and find distinct segments (singularly for every image in the stack):
        for z in range(final_segmentation.shape[0]):
            # Find connected segments:
            segm_relabeled = vigra.analysis.labelImageWithBackground(final_segmentation[z])

            # Find segment sizes:
            segment_sizes = get_map_of_segment_sizes_in_segmentation(segm_relabeled)

            # Delete segments that are smaller than the given threhsold:
            final_segmentation[z][segment_sizes < SIZE_THRESH] = 0

        # # Read raw data:
        # raw_path = os.path.join(raw_data_dir, rel_path).replace("_results.h5", ".h5")
        # raw = shp_utils.readHDF5(raw_path, "data")

        # Write new segmentation to file:
        file_out_path = full_path.replace("_results.h5", "_postProcSegm.h5")
        # shp_utils.writeHDF5(raw, file_out_path, "raw")
        final_segmentation = final_segmentation.astype('uint8')
        shp_utils.writeHDF5(final_segmentation, file_out_path, "data")

        # Write as .tif stack:
        final_segmentation[final_segmentation == 1] = 255
        tifffile.imwrite(file_out_path.replace('.h5', '.tif'), final_segmentation, photometric='minisblack', compress=4)
        print(rel_path, "written")

        # from shapeAnalysis.video_utils import save_segmentation_video
        # save_segmentation_video(raw, final_segmentation, file_out_path.replace(".h5", ".pdf"))
        # break

# shp_utils.check_dir_and_create(root_out_data_dir)

print("Done")

