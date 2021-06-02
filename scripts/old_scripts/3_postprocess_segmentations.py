import os
from shapeAnalysis import utils as shp_utils
import vigra
import numpy as np
import argparse
from nifty import tools as ntools
import tifffile


def get_map_of_segment_sizes_in_segmentation(segmentation):
    """
    Compute a size map of each segment in the given segmentation
    """
    node_sizes = np.bincount(segmentation.flatten())
    return ntools.mapFeaturesToLabelArray(segmentation, node_sizes[:, None], nb_threads=-1).squeeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str)
    parser.add_argument('--PROJECT_DIR', type=str)
    parser.add_argument('--size_thresh', type=int, default=40)
    parser.add_argument('--write_tif_images', type=bool, default=True)
    args = parser.parse_args()

    DATA_DIR = args.DATA_DIR
    PROJECT_DIR = args.PROJECT_DIR
    size_thresh = args.size_thresh
    write_tif_images = args.write_tif_images
    assert os.path.exists(PROJECT_DIR), "PROJECT_DIR not found: {}".format(PROJECT_DIR)

    segmentations_subdir = "segmentations"

    segm_dir = os.path.join(PROJECT_DIR, segmentations_subdir)
    paths_input_images = shp_utils.get_paths_input_files(segm_dir, ".h5")


    for full_path, rel_path in paths_input_images:
        if "_segmentation.h5" in full_path:
            # Load segmentation:
            # Label 1 --> foreground
            # Label 2 --> background
            final_segmentation = shp_utils.readHDF5(full_path, "exported_data")
            # Set background to zero, convert and get rid of last dimension:
            final_segmentation = final_segmentation.astype('uint32')[..., 0]
            final_segmentation[final_segmentation==2] = 0

            # Run connected components and find distinct segments (singularly for every timeframe):
            for z in range(final_segmentation.shape[0]):
                # Find connected segments:
                segm_relabeled = vigra.analysis.labelImageWithBackground(final_segmentation[z])

                # Find segment sizes:
                segment_sizes = get_map_of_segment_sizes_in_segmentation(segm_relabeled)

                # Delete segments that are smaller than the given threhsold:
                final_segmentation[z][segment_sizes < size_thresh] = 0

            # Write new segmentation to file:
            file_out_path = full_path.replace("_segmentation.h5", "_postProcSegm.h5")
            final_segmentation = final_segmentation.astype('uint8')
            shp_utils.writeHDF5(final_segmentation, file_out_path, "data")

            # Write as .tif stack:
            if write_tif_images:
                final_segmentation[final_segmentation == 1] = 255
                tifffile.imwrite(file_out_path.replace('.h5', '.tif'), final_segmentation, photometric='minisblack', compress=4)
            print(rel_path, "postprocessed!")




