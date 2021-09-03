"""MultiFileDataset

Contains the class definition for a distributed, multi-file, binary dataset.
"""
import numpy as np

def offset_slicer(slicer, offset, shape):
    """Offsets a slicer by `offset` while bounding by `shape` and preserving
    strides.

    Parameters
    ----------
    slicer: iterable of slice objects
        The original slices.
    offset: iterable of int
        The distance (in elements) to offset the slices. Offsets are added to
        the slice start.
    shape: iterable of int
        The maximum shape to allow. If a slice end point moves past the
        corresponding shape element, it is clipped.

    Notes
    -----
    When the stride of the slice is not 1, this is not a simple shift. Stride
    is properly respected when the slice start point moves below the origin.
    See examples.

    Examples
    --------
    >>> slicer = [slice(0, 5, 2)]
    >>> shape = (10,)

    Offset without hitting bounds.

    >>> offset_slicer(slicer, [1], shape)
    (slice(1, 6, 2),), (0,)

    Offset, moving past upper bound.

    >>> offset_slicer(slicer, [7], shape)
    (slice(7, 10, 2),), (0,)

    Offset, moving below lower bound.

    >>> offset_slicer(slicer, [-1], shape)
    (slice(1, 4, 2),), (1,)
    >>> offset_slicer(slicer, [-2], shape)
    (slice(0, 3, 2),), (1,)
    >>> offset_slicer(slicer, [-3], shape)
    (slice(1, 2, 2),), (2,)
    """
    new_slicer = []
    lower_offset = []
    for aslice, aoff, alen in zip(slicer, offset, shape):
        step = 1
        if aslice.step:
            step = aslice.step

        start = aslice.start + aoff
        if start < 0:
            new_start = start % step
            lower_offset.append((new_start - start) // step)
            start = new_start
        else:
            lower_offset.append(0)

        stop = min(alen, aslice.stop + aoff)
        new_slicer.append(slice(start, stop, step))
    return tuple(new_slicer), tuple(lower_offset)


class MultiFileDataset:
    """MultiFileDataset

    Memory maps a collection of raw binary files into a single array.

    Parameters
    ----------
    datafiles: array-like of str
        The files to map into, in topological order.
    fileshape: tuple
        The shape (in elements) of the dataset in each file.
    offset: int, optional
        The offset of the dataset in the file, in bytes.
    stride: int, optional
        The stride of the data as layed out in the file, in elements.
    ghosts: None or tuple, optional
        If present, the data in the file will have shape `fileshape + 2*ghosts`,
        but only the inner `fileshape` elements will be used. The border cells
        will be ignored.
    dtype: data-type, optional
        The data type of the file.
    """
    def __init__(self, datafiles, fileshape, offset=0, stride=1, ghosts=None,
                 dtype='float32', order='C'):
        self._files = np.atleast_1d(datafiles)
        self._shape = np.atleast_1d(fileshape).astype('int')
        self._offset = int(offset)
        self._stride = int(stride)
        self._order = order
        self._dtype = np.dtype(dtype)
        if ghosts is None:
            ghosts = (0,)*len(fileshape)
        self._ghosts = np.atleast_1d(ghosts).astype('int')

    def __repr__(self):
        return f'<MultiFileDataset, shape={self.shape}>'

    @property
    def files(self):
        """The array of underlying files."""
        return self._files

    @property
    def shape(self):
        """The shape of the combined dataset."""
        topology = self.files.shape
        return tuple([self._shape[i]*n for i, n in enumerate(topology)])

    @property
    def ndim(self):
        """Dimensionality of the combined dataset."""
        return len(self._shape)

    @property
    def size(self):
        """Total size in elements of the combined dataset."""
        return np.prod(self.shape)

    def _get_memmap(self, filename):
        """Construct a memory-mapped view into a file."""
        shape = self._shape + 2*self._ghosts
        buflen = np.prod(shape)*self._stride - (self._stride-1)
        databuf = np.memmap(filename,
                            mode='r',
                            dtype=self._dtype,
                            shape=(buflen,),
                            offset=self._offset)
        # Apply striding.
        databuf = databuf[::self._stride].reshape(shape, order=self._order)

        # Remove ghosts.
        ghost_slicer = []
        for ghost in self._ghosts:
            if ghost == 0:
                ghost_slicer.append(slice(None))
            else:
                ghost_slicer.append(slice(ghost, -ghost))
        databuf = databuf[tuple(ghost_slicer)]

        # Return valid data.
        return databuf

    def _gather_data(self, files, slicer):
        """Gather the data. We are only loading the required data into memory"""

        shape = [(aslice.stop-aslice.start)//aslice.step for aslice in slicer]
        data = np.zeros(shape, dtype=self._dtype)

        for findex in np.ndindex(files.shape):

            # Load the local block and slice off what we need.
            # At this point, local_slicer refers to the individual file data
            # not including ghost cells.
            #
            #  +---------------------------+
            #  |              #############|
            #  |              #############|
            #  |              #############|       ## = selected data
            #  |              #############|        0 = Data origin
            #  |              #############|
            #  |                           |
            #  |                           |
            #  0---------------------------+


            # Move slicer to file origin.
            local_slicer, offset = offset_slicer(slicer, 
                                                 -(findex*self._shape),
                                                 self._shape)

            # Get data.
            local_data = self._get_memmap(files[findex])[local_slicer]

            # Insert the block into the output data.
            local_slicer = tuple([slice(off, off+size) for off, size
                                  in zip(offset, local_data.shape)])
            data[local_slicer] = local_data

        # And return.
        return data

    def __getitem__(self, slicer):
        """Load the needed data and slice the dataset."""

        # Single axis slicing.
        if not isinstance(slicer, tuple):
            slicer = (slicer, Ellipsis)

        # Expand Ellipsis
        slicer = list(slicer)
        if Ellipsis in slicer:
            index = slicer.index(Ellipsis)
            slicer.remove(Ellipsis)
            for i in range(self.ndim - len(slicer)):
                slicer.insert(index, slice(None))

        # Construct the file slicer and offsets.
        offset = []
        fileslicer = []
        squeeze_axes = []
        shape = self.shape
        for i, axslice in enumerate(slicer):

            if isinstance(axslice, int):
                slicer[i] = slice(axslice, axslice+1)
                axslice = slicer[i]
                squeeze_axes.append(i)
            elif not isinstance(axslice, slice):
                raise KeyError('Only slices and single indicies are supported.'
                               ' Try reading a slice first, then use fancy'
                               ' indexing.')

            # Expand None into indicies.
            start, stop, stride = axslice.indices(shape[i])
            slicer[i] = slice(start, stop, stride)

            # Find which files we need to access.
            start = start//self._shape[i]
            stop = 1 + (stop-1)//self._shape[i]
            offset.append(-start*self._shape[i])
            fileslicer.append(slice(start, stop))

        slicer, _ = offset_slicer(slicer, offset, self.shape)
        files = self.files[tuple(fileslicer)]

        # At this point, fileslicer prunes off files that don't contain needed
        # data, while slicer has been shifted to account for the missing files
        #
        #  +------+------+------+------+
        #  |  XX  |  XX  |  XX  |  XX  |
        #  +------+------+------+------+       XX = pruned file
        #  |  XX  |   ###|##### |  XX  |       ## = selected data
        #  +------+---###+#####-+------+        0 = New slicer origin
        #  |  XX  |   ###|##### |  XX  |
        #  +------0------+------+------+
        #  |  XX  |  XX  |  XX  |  XX  |
        #  +------+------+------+------+

        # Now we just need to gather the data.
        data = self._gather_data(files, slicer)
        if squeeze_axes:
            data = data.squeeze(axis=tuple(squeeze_axes))
        return data
