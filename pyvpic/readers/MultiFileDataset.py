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
    [slice(1, 6, 2)]

    Offset, moving past upper bound.

    >>> offset_slicer(slicer, [7], shape)
    [slice(7, 10, 2)]

    Offset, moving below lower bound.

    >>> offset_slicer(slicer, [-1], shape)
    [slice(1, 4, 2)]
    >>> offset_slicer(slicer, [-2], shape)
    [slice(0, 3, 2)]
    """
    new_slicer = []
    for aslice, aoff, alen in zip(slicer, offset, shape):
        step = 1
        if aslice.step:
            step = aslice.step

        start = aslice.start + aoff
        if start < 0:
            start = start % step

        stop = min(alen, aslice.stop + aoff)
        new_slicer.append(slice(start, stop, step))
    return tuple(new_slicer)

class MultiFileDataset:
    """MultiFileDataset

    Memory maps a collection of raw binary files into a single array.

    Parameters
    ----------
    datafiles: array-like of str, N-dimensions
        The files to map into, in topological order.
    fileshape: iterable (N,)
        The shape (in elements) of the dataset in each file. Each element can
        be either an int or array of ints describing the size of each dimension.
    offset: int or callable, optional
        The offset of the dataset in the file, in bytes. If a callable,
        signiture is offset(shape).
    stride: int, optional
        The stride of the data as layed out in the file, in elements.
    ghosts: None or tuple, optional
        If present, the data in the file will have shape `fileshape + 2*ghosts`,
        but only the inner `fileshape` elements will be used. The border cells
        will be ignored.
    dtype: data-type, optional
        The data type of the file.
    order: 'C' or 'F'
        The ordering of data.
    """
    def __init__(self, datafiles, fileshape, offset=0, stride=1, ghosts=None,
                 dtype='float32', order='C'):

        # Offset callbacks.
        if not callable(offset):
            self._offset = lambda shape: int(offset)
        else:
            self._offset = offset

        # Layour information
        self._stride = int(stride)
        self._order = order
        self._dtype = np.dtype(dtype)
        if ghosts is None:
            ghosts = (0,)*len(fileshape)
        self._ghosts = np.atleast_1d(ghosts).astype('int')

        # Store files.
        self._files = np.atleast_1d(datafiles)

        # Store file shapes and origins
        if isinstance(fileshape[0], (int, float)):
            fileshape = [int(x) * np.ones(self._files.shape[i], dtype='int')
                         for i, x in enumerate(fileshape)]

        # Store file information
        self._fileshape = fileshape
        self._fileorigin = [np.cumsum(shape)-shape[0] for shape in fileshape]
        self._shape = np.asarray([np.sum(x) for x in fileshape]).astype('int')

    @property
    def files(self):
        """The array of underlying files."""
        return self._files

    @property
    def shape(self):
        """The shape of the combined dataset."""
        return self._shape

    @property
    def ndim(self):
        """Dimensionality of the combined dataset."""
        return len(self._shape)

    @property
    def size(self):
        """Total size in elements of the combined dataset."""
        return np.prod(self.shape)

    def __repr__(self):
        shape_str = ", ".join(map(str, self.shape))
        return f'<MultiFileDataset, shape=({shape_str}), order={self._order}>'

    def _gather_data(self, slicer):
        """Gather the data. We are only loading the required data into memory.

        Parameters
        ----------
        slicer : iterable of slice
            The global data slicer.

        Returns
        -------
        data : ndarray
            The selected data.
        """

        # Load slicer information
        start = [aslice.start for aslice in slicer]
        shape = [(aslice.stop-aslice.start)//aslice.step for aslice in slicer]

        # First find the files that contain the data.
        file_slicer = []
        file_origin = []
        file_shape = []

        for shapes, origins, aslice in zip(self._fileshape, self._fileorigin, slicer):

            axis_start = np.searchsorted(origins, aslice.start, 'right')-1
            axis_stop = np.searchsorted(origins, aslice.stop-1, 'right')
            file_axis_slice = slice(axis_start, axis_stop)

            file_slicer.append(file_axis_slice)
            file_origin.append(origins[file_axis_slice])
            file_shape.append(shapes[file_axis_slice])

        files = self._files[tuple(file_slicer)]

        # Now load the data from all the files.
        data = np.zeros(shape, dtype=self._dtype)
        for findex in np.ndindex(files.shape):

            # Get the local fileshape
            local_shape = [file_shape[i][findex[i]] for i in range(self.ndim)]
            local_shape = np.atleast_1d(local_shape).astype('int')

            # Find the local origin
            local_origin = [file_origin[i][findex[i]] for i in range(self.ndim)]
            local_origin = np.atleast_1d(local_origin).astype('int')

            # Move slicer to file origin.
            local_slicer = offset_slicer(slicer, -local_origin, local_shape)

            # Get data.
            local_data = self._get_memmap(files[findex], local_shape)[local_slicer]

            # Find where this block belongs globally.
            offset = [local_origin[i] + lslice.start - start[i]
                      for i, lslice in enumerate(local_slicer)]

            # Insert the block into the output data.
            local_slicer = tuple([slice(off, off+size) for off, size
                                  in zip(offset, local_data.shape)])
            data[local_slicer] = local_data

        # And return.
        return data

    def _get_memmap(self, filename, local_shape):
        """Construct a memory-mapped view into a file."""
        shape = local_shape + 2*self._ghosts
        buflen = np.prod(shape)*self._stride - (self._stride-1)
        databuf = np.memmap(filename,
                            mode='r',
                            dtype=self._dtype,
                            shape=(buflen,),
                            offset=int(self._offset(shape)))
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

        # Now we just need to gather the data.
        data = self._gather_data(slicer)
        if squeeze_axes:
            data = data.squeeze(axis=tuple(squeeze_axes))
        return data
