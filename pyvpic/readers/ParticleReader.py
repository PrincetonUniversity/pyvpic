import os
import re
import time
import numpy as np
from .BaseReader import BaseReader
from .MultiFileDataset import MultiFileDataset, offset_slicer
from .VPICFile import VPICFile, reconstruct_domain

PARTICLE_DTYPE = np.dtype([
    ('dx', 'f4'), ('dy', 'f4'), ('dz', 'f4'), ('i', 'i4'),
    ('ux', 'f4'), ('uy', 'f4'), ('uz', 'f4'), ('w', 'f4')
])

def offsets_from_counts(counts, order='C'):
    """Construct buffer offsets from bin counts."""

    offsets = np.roll(np.cumsum(counts.flatten(order=order)), 1)
    offsets[0] = 0
    return offsets.reshape(counts.shape, order=order)


class ParticleCache:

    def __init__(self, particles, bins, counts, offsets, order='F'):
        """A cached array of particles."""

        if particles.dtype != PARTICLE_DTYPE:
            raise ValueError("Input data is not an array of particles.")

        self.particles = particles
        self.bins = bins
        self.counts = counts
        self.offsets = offsets
        self._order = order


    @property
    def shape(self):
        """Shape of the cached region."""
        return self.bins.shape


    def __len__(self):
        """Number of cached particles."""
        return len(self.particles)


    def __repr__(self):
        return f'<ParticleCache, shape={self.shape}, length={len(self)}>'


    def squeeze(self, axis=None):
        """Squeeze the bins, counts, and offsets."""

        return ParticleCache(
            self.particles,
            np.squeeze(self.bins, axis=axis),
            np.squeeze(self.counts, axis=axis),
            np.squeeze(self.offsets, axis=axis),
            order=self._order
        )


    def __getitem__(self, slicer):
        """Retrieve particles from a subset of cells."""

        # Turn slices into bins
        bins = self.bins[slicer]
        counts = self.counts[slicer]
        offsets = self.offsets[slicer]

        # Loop over voxels to get the subset.
        if len(bins) == 0:

            subset = np.zeros(0, dtype=PARTICLE_DTYPE)

        else:

            subset = np.concatenate([
                self.particles[off : off + count]
                for off, count in zip(offsets.flatten(order=self._order),
                                      counts.flatten(order=self._order))
            ])

        # Get the subset offsets.
        offsets = offsets_from_counts(counts, order=self._order)

        # Return a new array
        return ParticleCache(subset, bins, counts, offsets, order=self._order)


class ParticleMultiFileDataset(MultiFileDataset):
    """ParticleMultiFileDataset

    Memory maps a collection of raw particle files into a single array.

    Parameters
    ----------
    datafiles: array-like of str
        The files to map into, in topological order.
    fileshape: tuple
        The shape (in elements) of the dataset in each file.
    ghosts: None or tuple, optional
        If present, the data in the file will have shape `fileshape + 2*ghosts`,
        but only the inner `fileshape` elements will be used. The border cells
        will be ignored.
    """

    def __init__(self, datafiles, fileshape, order='C', presorted=False):

        self._presorted = presorted

        if self._presorted:
            self._cache = {}

        super().__init__(datafiles, fileshape, order=order,
                         dtype=PARTICLE_DTYPE, ghosts=(1,1,1))


    def __repr__(self):
        return f'<ParticleMultiFileDataset, shape={self.shape}>'


    def _get_filedata(self, filename, slicer):
        """Retrieves particle data from a file."""

        vfile = VPICFile(filename)
        header = vfile.header
        array_info = vfile.read_array_info()

        if array_info['element_size'] != self._dtype.itemsize:
            raise IOError('Invalid particle size detected.')

        # Update shape to match requested ordering.
        shape = np.array(header['grid']['shape'])
        if header['grid']['order'] != self._order:
            shape = shape[::-1]

        # Memmap the data.
        data = np.memmap(filename,
                         mode='r',
                         dtype=self._dtype,
                         shape=array_info['shape'],
                         offset=array_info['offset'],
                         order=array_info['order'])

        # Remove ghosts.
        ghost_slicer = []
        for ghost in self._ghosts:
            if ghost == 0:
                ghost_slicer.append(slice(None))
            else:
                ghost_slicer.append(slice(ghost, -ghost))

        # Find bins to load.
        num_voxels = np.prod(shape + 2*self._ghosts)
        bins = np.arange(num_voxels).reshape(shape + 2*self._ghosts, order=self._order)
        bins = bins[tuple(ghost_slicer)][slicer]

        # Indirectly sort and load particles.
        # No caching if particles are not presorted because this
        # requires caching the permute_vector which is Nparticles long.
        if not self._presorted:
            
            permute_vector = np.argsort(data['i'], kind='stable')
            bincounts = np.bincount(data['i'], minlength=num_voxels)
            data = np.concatenate([
                data[permute_vector[offsets[bin] : offsets[bin] + bincounts[bin]]]
                for bin in bins.flatten(order=self._order)
            ])

        # Load data using either cached counts, or reconstructing counts.
        else:

            # Load from the cache.
            if filename in self._cache:
                bincounts = self._cache[filename]['counts']

            # Construct and cache
            else:
                bincounts = np.bincount(data['i'], minlength=num_voxels)
                self._cache[filename] = {'counts': bincounts}

            # Builds offsets and load the particles.
            offsets = offsets_from_counts(bincounts, order=self._order)
            data = np.concatenate([
                data[offsets[bin] : offsets[bin] + bincounts[bin]]
                for bin in bins.flatten(order=self._order)
            ])

        # New offsets and counts.
        bincounts = bincounts[bins]
        offsets = offsets_from_counts(bincounts, order=self._order)

        # Adjust to global index.
        voxel_offset = header['rank']*num_voxels
        data['i'] += voxel_offset
        bins += voxel_offset

        return data, bins, bincounts, offsets


    def _gather_data(self, files, slicer):
        """Gather the data. We are only loading the required data into memory"""

        shape = [(aslice.stop-aslice.start)//aslice.step for aslice in slicer]

        particle_data = np.zeros(0, dtype=self._dtype)
        particle_bins = np.zeros(shape, dtype='int')
        particle_counts = np.zeros(shape, dtype='int')
        particle_offsets = np.zeros(shape, dtype='int')

        for findex in np.ndindex(files.shape):

            # Move slicer to file origin.
            local_slicer, offset = offset_slicer(slicer, 
                                                 -(findex*self._shape),
                                                 self._shape)

            # Get data.
            ldata, lbins, lcounts, loffsets = self._get_filedata(files[findex], local_slicer)

            # Update offsets and append data
            loffsets += len(particle_data)
            particle_data = np.concatenate([particle_data, ldata])

            # Insert the block into the output data.
            local_slicer = tuple([slice(off, off+size) for off, size
                                  in zip(offset, lbins.shape)])

            particle_bins[local_slicer] = lbins
            particle_counts[local_slicer] = lcounts
            particle_offsets[local_slicer] = loffsets

        # And return.
        return ParticleCache(particle_data,
                             particle_bins,
                             particle_counts,
                             particle_offsets)


class ParticleReader:

    def __init__(self, prefix, order='F', presorted=False):

        file_pattern = prefix + '.{step}.{rank}'
        self._order = order
        grid, datafiles, steps, shapes = reconstruct_domain(file_pattern, order=order)

        self._datafiles = datafiles
        self._grid = grid
        self._steps = steps
        self._fileshape = shapes
        self._presorted = presorted

        # Allows caching of one particle dataset for better performance.
        self._selected_index = -1
        self._dataset = None


    @property
    def topology(self):
        """The domain decomposition used."""
        if self._order == 'F':
            return self._datafiles.shape[:-1]
        return self._datafiles.shape[1:]


    @property
    def datasets(self):
        """The available datasets."""
        return ['particles']


    def get_grid(self):
        """Get the 4D grid for the particles, defined on cell centers."""
        return self._grid


    def get_timesteps(self, dataset):
        """Get the available timesteps for the particles."""
        return self._steps


    def __getitem__(self, slicer):

        if isinstance(slicer, tuple):
            step_index = slicer[0]
            slicer = slicer[1:]
        else:
            step_index = slicer
            slicer = None

        if not isinstance(step_index, int):
            raise NotImplementedError('Only single time indexing is supported.')

        if step_index != self._selected_index:

            self._selected_index = step_index
            self._dataset = ParticleMultiFileDataset(self._datafiles[step_index],
                                                     self._fileshape,
                                                     order=self._order,
                                                     presorted=self._presorted)

        if slicer is not None:
            return self._dataset[slicer]
        return self._dataset