import numpy as np
import yaml
import os
import h5py
import vigra

from scipy.ndimage import zoom


def get_paths_input_files(root_data_dir, file_extension=".tif"):
    paths_input_files = []
    for (dirpath, dirnames, filenames) in os.walk(root_data_dir):
        for filename in filenames:
            if filename.endswith(file_extension) and not filename.startswith("."):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, start=root_data_dir)
                paths_input_files.append((full_path, rel_path))
    return paths_input_files




def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return data_slice
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return tuple(slices)

# Yaml to dict reader
def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f, Loader=yaml.FullLoader)
    return readict

def check_dir_and_create(directory):
    '''
    if the directory does not exist, create it
    '''
    folder_exists = os.path.exists(directory)
    if not folder_exists:
        os.makedirs(directory)
    return folder_exists





def readHDF5(path,
             inner_path,
             crop_slice=None,
             dtype=None,
             ds_factor=None,
             ds_order=3,
             run_connected_components=False,
             ):
    if isinstance(crop_slice, str):
        crop_slice = parse_data_slice(crop_slice)
    elif crop_slice is not None:
        assert isinstance(crop_slice, tuple), "Crop slice not recognized"
        assert all([isinstance(sl, slice) for sl in crop_slice]), "Crop slice not recognized"
    else:
        crop_slice = slice(None)
    with h5py.File(path, 'r') as f:
        output = f[inner_path][crop_slice]

    if run_connected_components:
        assert output.dtype in [np.dtype("uint32")]
        assert output.ndim == 3 or output.ndim == 2
        output = vigra.analysis.labelVolumeWithBackground(output.astype('uint32'))
    if dtype is not None:
        output = output.astype(dtype)

    if ds_factor is not None:
        assert isinstance(ds_factor, (list, tuple))
        assert output.ndim == len(ds_factor)
        output = zoom(output, tuple(1./fct for fct in ds_factor), order=ds_order)

    return output

def readHDF5_from_volume_config(
        sample,
        path,
        inner_path,
        crop_slice=None,
        dtype=None,
        ds_factor=None,
        ds_order=3,
        run_connected_components=False,
        ):
    if isinstance(path, dict):
        if sample not in path:
            sample = eval(sample)
            assert sample in path
    path = path[sample] if isinstance(path, dict) else path
    inner_path = inner_path[sample] if isinstance(inner_path, dict) else inner_path
    crop_slice = crop_slice[sample] if isinstance(crop_slice, dict) else crop_slice
    dtype = dtype[sample] if isinstance(dtype, dict) else dtype
    return readHDF5(path, inner_path, crop_slice, dtype, ds_factor, ds_order, run_connected_components)

def writeHDF5(data, path, inner_path, compression='gzip'):
    if os.path.exists(path):
        write_mode = 'r+'
    else:
        write_mode = 'w'
    with h5py.File(path, write_mode) as f:
        if inner_path in f:
            del f[inner_path]
        f.create_dataset(inner_path, data=data, compression=compression)


def get_hdf5_inner_paths(path, inner_path=None):
    with h5py.File(path, 'r') as f:
        if inner_path is None:
            datasets = [dt for dt in f]
        else:
            datasets = [dt for dt in f[inner_path]]
    return datasets




