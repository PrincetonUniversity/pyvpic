import os
import struct
import fnmatch
import numpy as np


class VPICFile:

    VPIC_HEADER = '<bbbbbHIfdiiiiiiffffffffffiiif'
    HEADER_SIZE = struct.calcsize(VPIC_HEADER)

    def __init__(self, filename):

        self._filename = filename
        self._header = self.read_file_header()


    @property
    def header(self):
        """The file header contents."""
        return self._header


    def read_array_info(self, offset=HEADER_SIZE):
        """Reads a VPIC array header."""

        int_size = struct.calcsize('i')

        with open(self._filename, 'rb') as stream:
            stream.seek(offset)

            # Read primitive data size.
            data = stream.read(int_size)
            dsize = struct.unpack('i', data)[0]

            # Read dimensions
            data = stream.read(int_size)
            ndim = struct.unpack('i', data)[0]

            shape = []
            for _ in range(ndim):
                data = stream.read(int_size)
                shape.append(struct.unpack('i', data)[0])

            # Calculate new offset.
            offset = stream.tell()

        return {
            'offset': offset,
            'ndim': ndim,
            'element_size': dsize,
            'shape': tuple(shape),
            'order': 'F'
        }


    def read_file_header(self):
        """Validates a VPIC header file, and returns relavent information."""

        # Read the header.
        with open(self._filename, 'rb') as stream:
            header = stream.read(VPICFile.HEADER_SIZE)
        header = struct.unpack(VPICFile.VPIC_HEADER, header)

        # Now validate.
        if (
            header[0] != 8 or
            header[1] != 2 or
            header[2] != 4 or
            header[3] != 4 or
            header[4] != 8
        ):
            raise IOError('Non standard primitive sizes detected.')

        if (
            header[5] != 0xcafe or
            header[6] != 0xdeadbeef or
            header[7] != 1.0 or
            header[8] != 1.0
        ):
            raise IOError('Constant validation failed.')

        if (
            header[9] != 0
        ):
            raise IOError('Unknown VPIC version type.')

        # Return header contents
        return {
            'dump_type': header[10],
            'step': header[11],
            'rank': header[25],
            'nproc': header[26],
            'offset': VPICFile.HEADER_SIZE,
            'grid': {
                'start': tuple(header[19:22]),
                'delta': tuple(header[16:19]),
                'shape': tuple(header[12:15]),
                'order': 'F',
                'dt': header[15],
                'cvac': header[22],
                'eps0': header[23],
                'damp': header[24],
            },
            'species': {
                'id': header[27],
                'q_m': header[28]
            }
        }


def reconstruct_domain(file_pattern, order='F'):
    """Reconstructs a rectilinear domain decomposition from output files."""

    # Rank 0 must exist.
    time_pattern = os.path.basename(file_pattern).format(rank=0, step='*')
    directory = os.path.dirname(file_pattern)
    time = []
    steps = []

    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, time_pattern):
            header = VPICFile(os.path.join(directory, filename)).header
            nproc = header['nproc']
            time.append(header['step']*header['grid']['dt'])
            steps.append(header['step'])

    time = np.sort(time)
    steps = np.sort(steps)

    # Load ranks.
    x0 = np.zeros((nproc, 3))
    dx = np.zeros((nproc, 3))
    nx = np.zeros((nproc, 3))
    for rank in range(nproc):
        header = VPICFile(file_pattern.format(rank=rank, step=steps[0])).header
        x0[rank] = header['grid']['start']
        dx[rank] = header['grid']['delta']
        nx[rank] = header['grid']['shape']

    # Find unique origins.
    gnx = len(np.unique(x0[:,0]))
    gny = len(np.unique(x0[:,1]))
    gnz = len(np.unique(x0[:,2]))

    # Check for rectilinear.
    if gnx*gny*gnz != nproc:
        raise IOError('Domain is not rectilinear.')

    # Construct the global grid.
    def local_grid(rank, axis):
        return x0[rank, axis] + (0.5 + np.arange(nx[rank, axis]))*dx[rank, axis]

    x = np.concatenate([local_grid(ix, 0) for ix in range(gnx)])
    y = np.concatenate([local_grid(iy*gnx, 1) for iy in range(gny)])
    z = np.concatenate([local_grid(iz*gny*gnx, 2) for iz in range(gnz)])

    # Reorder datafiles and grid.
    datafiles = np.array([file_pattern.format(rank=rank, step=step)
                          for step in steps for rank in range(nproc)])

    # If we have a regular decomposition, simplify the shape.
    unique_shape = np.unique(nx, axis=0)
    if unique_shape.size == 3:
        shapes = tuple(unique_shape[0].astype('int'))
    else:
        raise NotImplementedError('Only uniform domain decompositions are supported.')

    if order == 'F':
        datafiles = datafiles.reshape((gnx, gny, gnz, len(steps)), order='F')
        grid = (x, y, z, time)
    else:
        datafiles = datafiles.reshape((len(steps), gnz, gny, gnx), order='C')
        grid = (time, z, y, x)
        shapes = shapes[::-1]

    # Return.
    return grid, datafiles, steps, shapes




