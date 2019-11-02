""" GDAReader
Module contains the base class for reading GDA files.
"""
import os
import re
import struct
import functools
import h5py 
import numpy as np
from .BaseReader import BaseReader

def _dset_attr(f):
    @property
    @functools.wraps(f)
    def inner(self):
        with h5py.File(self.filename, 'r') as h5file:
            return getattr(h5file[self.path], f.__name__)
    return inner

def _dset_func(f):
    @functools.wraps(f)
    def inner(self, *args, **kwargs):
        with h5py.File(self.filename, 'r') as h5file:
            return getattr(h5file[self.path], f.__name__)(*args, **kwargs)
    return inner

class H5DatasetWrapper():
    def __init__(self, filename, path, order='C'):
        self.filename = filename
        self.path = path
        self.order = order

        if order != 'C':
            raise NotImplementedError('Only "C" ordering currently supported.')

    @_dset_attr
    def name(): pass

    @_dset_attr
    def ndim(): pass

    @_dset_attr
    def shape(): pass

    @_dset_attr
    def size(): pass

    @_dset_func
    def __len__(): pass

    @_dset_func
    def __getitem__(): pass

    @_dset_func
    def __str__(): pass

    @_dset_func
    def __repr__(): pass


class H5Reader(BaseReader):
    """ Reads HDF5 output files. """

    def __init__(self, filename, order='C',**kwargs):
        super().__init__()
        self._order = order
        self._filename = filename
        self.read_datasets()

    def read_datasets(self):
        """ Read all the datasets in the selected file. """
        self._datasets = []
        
        def check_item(name, obj):
            """Callback to check if item is a dataset."""
            if 'grid' not in name.split('/'):
                if isinstance(obj, h5py.Dataset):
                    self._datasets.append(name)
                elif isinstance(obj, h5py.Group):
                    for next_name, next_obj in obj.items():
                        check_item(f'{name}/{next_name}', next_obj)

        # NOTE: visit/visititems is inexplicably slow on some
        # files/filesystems.
        #with h5py.File(self._filename, 'r') as h5file:
        #    h5file.visititems(check_item)

        with h5py.File(self._filename, 'r') as h5file:
            for item in h5file.items():
                check_item(*item)



    def get_grid(self, dataset):
        """Get the 4D grid for the dataset."""

        # Check for a valid dataset.
        if dataset not in self._datasets:
            raise KeyError(f'Unknown dataset "{dataset}"')
            
        # Find the best grid
        with h5py.File(self._filename, 'r') as h5file:

            dset = h5file[dataset]
            shape = dset.shape

            while dset != h5file:

                dset = dset.parent
                if ('grid' in dset and 
                    't' in dset['grid'] and 
                    'x' in dset['grid'] and
                    'y' in dset['grid'] and 
                    'z' in dset['grid']):

                    dset = dset['grid']
                    grid = [dset[x][:] for x in ['t','z','y','x']]                    
                    break

            else:
                # Default grid
                grid = [np.arange(i) for i in shape]
                
        # Allow column-major
        if self._order == 'F':
            grid = grid[::-1]

        return grid

    def __getitem__(self, dataset):
        """Open a dataset. Datasets are returned as a wrapper around h5py 
        Dataset objects.  This delays loading of data and speeds up access 
        to large datasets, while also preventing HDF5 files from remaining
        open."""

        # Build and return the dataset.
        return H5DatasetWrapper(self._filename, dataset, order=self._order)

    @property
    def datasets(self):
        """A list of all datasets available."""
        return self._datasets
