"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from scipy.io import loadmat
import os

# supported spero file extensions
spero_file_extensions = ('.mat')  # all extensions: ('.hdr', '.mat', '.dcf')


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, path to folder or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    if os.path.isdir(path):
        # check if folder provided has at least one valid file
        for fname in os.listdir(path):
            if fname.endswith(spero_file_extensions):
                break

    # if we know we cannot read the file, we immediately return None.
    elif not path.endswith(spero_file_extensions):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, path to folder or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """

    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path

    spero_file_sets = []
    for _path in paths:
        if os.path.isdir(_path):
            spero_file_sets += [[f'{_path}/{file}' for file in os.listdir(_path)
                                 if file.endswith(spero_file_extensions)]]
        else:
            spero_file_sets += [[_path]]
    
    # load all files into arrays
    napari_layers = []
    for spero_file_set in spero_file_sets:
        # here I did not flat files in folders in order to
        # allow for grouping as soon as implemented
        for spero_file in spero_file_set:
            matlab_dict = loadmat(spero_file)
            image_data = matlab_dict.pop('r').reshape((480, 480, -1)).transpose(2, 1, 0)
            image_data = np.power(10, -image_data)

            # Define layer kwargs
            # find translation form stage
            stage_x_data = matlab_dict['stage_X_um'][0, 0]
            stage_y_data = matlab_dict['stage_Y_um'][0, 0]

            # find wavelength for defining 1st dimension
            wavenumbers = matlab_dict['wn'][:, 0]

            # ensures same orientation as in ChemVision software
            stage_data = (wavenumbers[0], -stage_y_data, stage_x_data)

            # define default scale as low mag objective
            # maybe there is a smarter way by comparing
            # stage positions then multiple files are loaded
            scale = (wavenumbers[1]-wavenumbers[0], 2025/480, 2025/480)
            
            # min max as contrast_limit
            contrast_limits = (np.min(image_data), np.max(image_data))

            # Assamble layer parameter
            # Add all but image data as metadata, removed by pop
            layer_params = {'scale': scale,
                            'translate': stage_data,
                            'contrast_limits': contrast_limits,
                            'metadata': matlab_dict}

            # Define layer type
            layer_type = "image"

            # Assamble napari layers
            napari_layers += [(image_data, layer_params, layer_type)]
            
    # apply global contrast limits
    contrast_limits_global = (min(napari_layers, key=lambda item: item[1]['contrast_limits'][0])[1]['contrast_limits'][0],
                              max(napari_layers, key=lambda item: item[1]['contrast_limits'][1])[1]['contrast_limits'][1])
                           
    contrast_limits_global = tuple(contrast_limits_global + np.diff(contrast_limits_global) *[-0.15, 0.15])

    for nl in napari_layers:
        nl[1]['contrast_limits'] = contrast_limits_global
    
    return napari_layers


def tolerance_clustering(data, tolerance):
    # Sort the data
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    # Calculate differences between adjacent elements and round within tolerance
    diff = np.diff(sorted_data)
    clustered_diff = np.where(diff < tolerance, 0, diff)

    # Extend the size of the differences array to match the size of the original data
    clustered_diff = np.concatenate((sorted_data[0:1], clustered_diff))

    # Create the clustered data using cumulative addition
    clustered_data = np.cumsum(clustered_diff)

    # Revert the sorting to the original order
    clustered_data = clustered_data[np.argsort(sorted_indices)]

    return clustered_data


def combine_mosaic(matlab_dict_set):
    image_data = [matlab_dict.pop('r').reshape((480, 480, -1)) for matlab_dict in matlab_dict_set]
    stage_x_data = np.array([matlab_dict.pop('stage_X_um')[0, 0] for matlab_dict in matlab_dict_set])
    stage_y_data = np.array([matlab_dict.pop('stage_Y_um')[0, 0] for matlab_dict in matlab_dict_set])
    wavenumbers = matlab_dict_set[0]['wn']

    # Trivial cases
    if len(image_data) == 0:
        return None

    if len(image_data) == 1:
        out = image_data[0]

    else:

        # bin close values by clustering with a tolerance
        stage_x_rounded = tolerance_clustering(stage_x_data, 5)
        stage_y_rounded = tolerance_clustering(stage_y_data, 5)

        # determine image order based on binned coordinates
        image_order = list(np.argsort(stage_x_rounded*10+stage_y_rounded))

        # Extract number of measurements in x and y
        nr_columns = np.unique(stage_x_rounded).shape[0]
        nr_rows = np.unique(stage_y_rounded).shape[0]

        # create empty array with correct dimensions for final mosaic
        mosaic_shape = (nr_columns * 480, nr_rows * 480, len(wavenumbers))
        out = np.zeros(mosaic_shape)

        for column in range(nr_columns):
            for row in range(nr_rows):
                out[column * 480: (column + 1) * 480, row * 480: (row + 1) * 480] = image_data[image_order.pop()][::-1]
        out = out[::-1]

    out = out.transpose(2, 1, 0)
    return out
