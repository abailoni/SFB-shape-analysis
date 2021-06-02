import skimage.io
import os
from . import utils as shp_utils
import numpy as np

import vigra
from nifty import tools as ntools
import tifffile


def get_map_of_segment_sizes_in_segmentation(segmentation):
    """
    Compute a size map of each segment in the given segmentation
    """
    node_sizes = np.bincount(segmentation.flatten())
    return ntools.mapFeaturesToLabelArray(segmentation, node_sizes[:, None], nb_threads=-1).squeeze()



def pre_process_video(video):
    # Normalize betwen 0 and 1 and take median along time-dimension:
    video = video.astype('float32') / 255.
    median_values = np.median(video, axis=0)
    normalized_video = video - median_values

    # Re-normalize so that median value is fixed at value 128:
    mask1 = normalized_video < 0.
    mask2 = np.logical_not(mask1)
    normalized_video_1 = normalized_video / normalized_video.min()  # Array with values in [0, 1]
    normalized_video_2 = normalized_video / normalized_video.max()  # Array with values in [0, 1]
    normalized_video[mask1] = normalized_video_1[mask1] * (-128.)  # Array with values in [-128, 0]
    normalized_video[mask2] = normalized_video_2[mask2] * 127.  # Array with values in [0, 127]
    normalized_video += 128.
    normalized_video = normalized_video.astype('uint8')

    return normalized_video


def pre_process_all_videos(project_dir, all_video_names):
    for name_video in all_video_names:
        video_path = os.path.join(project_dir, "input_files", "{}.h5".format(name_video))
        video = shp_utils.readHDF5(video_path, "data")
        print("Processing video {} with shape {}".format(name_video, video.shape))
        preprocessed = pre_process_video(video)

        # Writing:
        out_file = os.path.join(project_dir, "preprocessed", "{}.h5".format(name_video))
        out_dir = os.path.split(out_file)[0]
        shp_utils.check_dir_and_create(out_dir)
        shp_utils.writeHDF5(preprocessed, out_file, "data")

def convert_tif_images_to_hdf5_stack(input_dir, project_dir, all_video_names):
    for name_video in all_video_names:
        for (dirpath, dirnames, filenames) in os.walk(os.path.join(input_dir, name_video)):
            index = 1

            # Deduce how many frames the video has:
            max = 0
            for filename in filenames:
                if filename.endswith(".tif") and not filename.startswith("."):
                    # Deduce index:
                    index_frame = int(filename.replace(name_video, "").replace(".tif", ""))
                    if index_frame > max:
                        max = index_frame
                        # TODO: check shape 2D or 3D
                        full_path = os.path.join(dirpath, filename)
                        xy_shape = skimage.io.imread(full_path).shape
            assert max > 0, "No frames found for {}".format(name_video)
            print("Found {} frames in video {}".format(max, name_video))
            video = np.empty((max,) + xy_shape, dtype='uint8')
            for filename in filenames:
                if filename.endswith(".tif") and not filename.startswith("."):
                    index_frame = int(filename.replace(name_video, "").replace(".tif", ""))
                    full_path = os.path.join(dirpath, filename)
                    # TODO: check shape 2D or 3D
                    video[index_frame-1] = skimage.io.imread(full_path)

            # Write video in hdf5:
            out_dir = os.path.join(project_dir, "input_files")
            shp_utils.check_dir_and_create(out_dir)
            out_file = os.path.join(out_dir, "{}.h5".format(name_video))
            shp_utils.writeHDF5(video, out_file, "data")
            print(out_file, "written")


def run_ilastik_on_all_videos(project_dir, all_video_names, path_ilastik, path_ilastik_proj):
    # Create out-dir:
    file_out_dir = os.path.join(project_dir, "outputs_ilastik")
    shp_utils.check_dir_and_create(file_out_dir)

    for name_video in all_video_names:
        input_file_path = os.path.join(project_dir, "preprocessed", "{}.h5".format(name_video))
        assert os.path.exists(input_file_path)

        # Run ilastik in headless mode:
        command = "{} --headless --readonly --input_axes=tyx --project={} --export_source=\"Simple Segmentation\" --output_filename_format={}/{{nickname}}_segmentation.hdf5  --output_format=\"compressed hdf5\" --export_dtype=uint8 {}".format(
            path_ilastik, path_ilastik_proj, file_out_dir, input_file_path)
        stream = os.popen(command)
        output = stream.read()
        print(output)
        print("\n")


def post_process_videos(project_dir, all_video_names, size_thresh):
    # Create out-dir:
    file_out_dir = os.path.join(project_dir, "postprocessed_outputs")
    shp_utils.check_dir_and_create(file_out_dir)

    for name_video in all_video_names:
        # Load segmentation:
        path_input_segmentation = os.path.join(project_dir, "outputs_ilastik", "{}_segmentation.h5".format(name_video))
        # Label 1 --> foreground
        # Label 2 --> background
        final_segmentation = shp_utils.readHDF5(path_input_segmentation, "exported_data")
        # Set background to zero, convert and get rid of last dimension:
        final_segmentation = final_segmentation.astype('uint32')[..., 0]
        final_segmentation[final_segmentation == 2] = 0

        # Run connected components and find distinct segments (singularly for every timeframe):
        for z in range(final_segmentation.shape[0]):
            # Find connected segments:
            segm_relabeled = vigra.analysis.labelImageWithBackground(final_segmentation[z])

            # Find segment sizes:
            segment_sizes = get_map_of_segment_sizes_in_segmentation(segm_relabeled)

            # Delete segments that are smaller than the given threhsold:
            final_segmentation[z][segment_sizes < size_thresh] = 0

        # Write new segmentation to file:
        file_out_path = os.path.join(file_out_dir, "{}.tif".format(name_video))
        final_segmentation = final_segmentation.astype('uint8')

        final_segmentation[final_segmentation == 1] = 255
        tifffile.imwrite(file_out_path, final_segmentation, photometric='minisblack', compress=4)
        print(name_video, "postprocessed!")
