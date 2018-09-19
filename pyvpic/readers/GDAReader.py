""" GDAReader
Module contains the base class for reading GDA files.
"""
import os
import re
import struct
import numpy as np
from .BaseReader import BaseReader
from .MultiFileDataset import MultiFileDataset

class GDAReader(BaseReader):
    """ Reads GDA output files (brick of values).
    """

    def __init__(self, directory, access='r', order='C', recursive=True,
                 padding=None, **kwargs):
        super().__init__()
        self._access_mode = access
        self._order = order
        self._files = {}
        self.open_directory(directory, recursive=recursive)
        if padding is None:
            padding = self.check_gda_padding()
        self._padding = padding

    def check_gda_padding(self):
        """ Check if there is padding between timesteps. This is
        *not* robust and can fail under some corner cases."""
        if not self.datasets:
            return 0

        fname = self._files[self.datasets[0]]
        if not os.path.isfile(fname):
            return 0

        grid = self._get_file_grid(fname)
        filesize = os.stat(fname).st_size

        if self._order == 'F':
            nx, ny, nz, nt = [len(axis) for axis in grid]
        else:
            nt, nz, ny, nx = [len(axis) for axis in grid]

        padding = filesize/(4*nt) - nz*ny*nx
        if padding != int(padding):
            raise IOError("Unable to determine GDA padding.")

        return int(padding)

    def open_directory(self, directory, recursive=True):
        """ Opens a directory and finds all the GDA files within it. """
        self._files = {}

        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:

                # Ignore non-GDA files
                if not filename.endswith('.gda'):
                    continue

                # Remove timstamps if present.
                filename = re.sub(r'_\d+(?=\.gda)', '', filename)

                # Build some names.
                curpath = os.path.join(dirpath, filename)
                dataset = os.path.relpath(curpath, directory)[:-4]
                abspath = os.path.abspath(curpath)

                # Add to flat list of files.
                if dataset not in self._files:
                    self._files[dataset] = abspath

            if not recursive:
                break

    def _get_file_grid(self, filename):
        """Get the 4D grid for the GDA file. """

        # Find the info file.
        dirname = os.path.dirname(filename)
        if 'info' not in os.listdir(dirname):
            raise IOError("Cannot find 'info' file.")

        # Read the grid
        with open(os.path.join(dirname, 'info'), mode='rb') as info:
            data = struct.unpack('4x3i8x3f', info.read(36))
            nx, ny, nz, Lx, Ly, Lz = data

        # Compute the number of timesteps.
        if os.path.isfile(filename):
            filesize = os.stat(filename).st_size
            time = np.arange(filesize//(4*nx*ny*nz))

        else:
            prefix = os.path.basename(filename).strip('.gda')
            time = []
            for testfile in os.listdir(dirname):
                match = re.match(rf'{prefix}_(\d+)\.gda', testfile)
                if match:
                    time.append(int(match.group(1)))
            time.sort()
            time = np.atleast_1d(time).astype('int')


        # Compute the grid
        grid = [
            time,
            np.linspace(0, Lz, nz),
            np.linspace(0, Ly, ny),
            np.linspace(0, Lx, nx)
        ]

        # Allow column-major
        if self._order == 'F':
            grid = grid[::-1]

        return grid

    def get_grid(self, dataset):
        """Get the 4D grid for the dataset. GDA only supports a single
        grid per directory.

        The grid info file should be a binary file formatted as:
            4b - padding
            float32 - nx
            float32 - ny
            float32 - nz
            8b - padding
            float32 - Lx
            float32 - Ly
            float32 - Lz
        """
        # Check for a valid dataset.
        if dataset not in self._files:
            raise KeyError(f'Unknown dataset "{dataset}"')
        return self._get_file_grid(self._files[dataset])

    def __getitem__(self, dataset):
        """Open a dataset.

        Datasets are opened as a memory-mapped view into the file with the order
        and access mode set on class construction. With 'C' ordering, indicies
        are (time, z, y, x), and with 'F' ordering indicies are (x, y, z, time).
        """
        filename = self._files[dataset]
        grid = self.get_grid(dataset)
        shape = [len(axis) for axis in grid]

        # Single file, multiple timesteps.
        # Create a memmap into the file. Not we are doing some tricks here
        # to eliminiate the padding, but since it is a memmap it is all in
        # the view and not on the data itself.
        if os.path.isfile(filename):

            # F ordering
            if self._order == 'F':
                grid_size = np.prod(shape[:-1])
                dataset = np.memmap(filename,
                                    dtype='float32',
                                    mode=self._access_mode,
                                    shape=(self._padding + grid_size, shape[-1]),
                                    order='F'
                                   )
                return dataset[:grid_size, :].reshape(shape, order='F')

            # C ordering by default.
            grid_size = np.prod(shape[1:])
            dataset = np.memmap(filename,
                                dtype='float32',
                                mode=self._access_mode,
                                shape=(shape[0], self._padding + grid_size),
                                order='C'
                               )
            return dataset[:, :grid_size].reshape(shape, order='C')

        # Single file, single timestep.
        # Create a stack of memmaps into all the files.
        if self._order == 'C':
            shape[0] = 1
            timesteps = grid[0]
            new_shape = (-1, 1, 1, 1)
        else:
            shape[-1] = 1
            timesteps = grid[-1]
            new_shape = (1, 1, 1, -1)

        # Convert time indicies into filename
        prefix = filename.strip('.gda')
        datafiles = np.array([f'{prefix}_{step}.gda' for step in timesteps])
        datafiles = datafiles.reshape(new_shape)

        # Build and return the dataset.
        return MultiFileDataset(datafiles, shape, order=self._order)

    @property
    def datasets(self):
        """A list of all datasets available."""
        return list(self._files.keys())
