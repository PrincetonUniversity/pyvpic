""" data

Wraps the file-specifc readers and provides a uniform interface to
VPIC data, regardless of how the data is stored.
"""
import os
import pyvpic.readers as readers

def open(path, **kwargs):
    """Open a path and construct an appropriate data reader.

    Output data from VPIC can be stored in several different formats and this
    function is a convience wrapper for constructing an appropriate reader.
    Currently implemented readers are:

    GDAReader
        Reads GDA (brick of value) files including both single-file,
        multiple-timestep and single-file, single-timestep GDA files. To use
        the GDAReader, `path` should be a directory containing GDA files. By
        default directories are search recursively for GDA files. Each
        directory containing GDA files should also contain a binary `info` file
        that describes the dimensions of the datasets.

    FilePerRankReader
        Reads the raw file-per-rank outputs form VPIC. To use this reader,
        `path` should point to a VPIC header file (usually `global.vpc`). VPIC
        dumps may be either banded or interleaved, and this should be set using
        the `interleave` keyword.

    H5Reader
        Reads HDF5 file outputs form VPIC. To use this reader, `path` should 
        point to an HDF5 file.

    Parameters
    ----------
    path: string
        Path to a data source.
    kwargs: optional
        Reader specifc keyword arguments. See the individual readers for
        details.

    Returns
    -------
    reader: BaseReader
        An instance of a subclass of `BaseReader` which provides the datasets.

    """

    if os.path.isdir(path):
        return readers.GDAReader(path, **kwargs)

    elif path.endswith('.vpc'):
        return readers.FilePerRankReader(path, **kwargs)

    elif path.endswith('.h5'):
        return readers.H5Reader(path, **kwargs)

    raise IOError('Unable to determine reader for "{path}".')
