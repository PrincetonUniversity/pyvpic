import os
import re
import numpy as np
from .BaseReader import BaseReader
from .MultiFileDataset import MultiFileDataset

class FilePerRankReader(BaseReader):

    """Reads in the distributed datasets written by the new VPIC dump format.

    Parameters
    ----------
    header_file: path
        The VPIC header file, usually named `global.vpc`.
    order: {'C', 'F'}, optional
        The order of the arrays, either row-major, C-style, or column-major,
        Fortran-style. This does not affect the underlying layout in memory,
        only the python view. For C-ordering, axes are (time, z, y, x) while
        for F-ordering, axes are (x, y, z, time).
    interleave: bool, optional
        Whether the VPIC outputs are banded (`False`) or interleave-banded
        (`True`). This parameter is not stored in the header file, and must
        be correctly specified here otherwise the returned data will be
        meaningless.

    Attributes
    ----------
    VARNAMES: dict
        Translation from long to short names for datasets.
    topology: array of int
        The global topology of the datafiles (MPI domains).
    datasets: array of str
        The datasets available for reading.

    """


    VARNAMES = {
        # Field variables
        'electric field': 'e',
        'magnetic field': 'b',
        'free current field': 'j',
        'tca field': 'tca',
        'charge density': 'rho',
        'bound charge density': 'rhob',
        'electric field divergence error': 'dive',
        'magnetic field divergence error': 'divb',
        'edge material': 'edgemat_',
        'face material': 'facemat_',
        'node material': 'nodemat',
        'cell material': 'cellmat',

        # Hydro variables
        'current density': 'j',
        # 'charge density': 'rho',  NOTE: this is a duplicate of above.
        'momentum density': 'p',
        'kinetic energy density': 'ke',
        'stress tensor': 't'
    }

    def __init__(self, header_file, order='C', interleave=True, **kwargs):
        super().__init__()

        # Default parameters. These are updated by read_header.
        self._params = {
            'grid': {
                'delta_t': 1,
                'topology_z': 0,
                'topology_y': 0,
                'topology_x': 0
            },
            'field': {'dtype': np.dtype([])},
            'species': [],
            'basedir': ''
        }
        self._order = order
        self._interleave = interleave
        self.read_header(header_file)

        # Construct the topology array.
        self._topology = np.array([
            self._params['grid']['topology_z'],
            self._params['grid']['topology_y'],
            self._params['grid']['topology_x']
        ]).astype('int')
        if self._order == 'F':
            self._topology = self._topology[::-1]

    @property
    def topology(self):
        """The domain decomposition used."""
        return self._topology

    def read_header(self, header_file):
        """ Reads in the VPIC header file (usually named global.vpc).

        Parameters
        ----------
        filename: path
            Path to the VPIC header file.
        """

        # Initialize the parameters.
        self._params = {
            'grid': {},
            'field': {'dtype': []},
            'species': [],
            'basedir': os.path.dirname(os.path.abspath(header_file))
        }
        params = self._params

        # Read the header file.
        with open(header_file, 'r') as header:
            for line in header:

                # Read a parameter, we need to maintain capitals here.
                match = re.match(r'([a-zA-Z]+)_([_a-zA-Z]+) (.*)', line)
                if match:
                    group, key, values = match.groups()
                    group = group.lower()
                    key = key.lower()

                    values = values.split(' ')
                    try:
                        values = list(map(float, values))
                    except:
                        pass
                    if len(values) > 1:
                        values = np.array(values)
                    else:
                        values = values[0]

                    if group == 'species' or group == 'hydro':
                        params['species'][active_species][key] = values
                    elif group in params:
                        params[group][key] = values
                    else:
                        params[key] = values

                # Uppercase is ugly, get rid of it for everything else.
                line = line.lower()

                # Initialize the species.
                match = re.match(r'num_output_species (\d+)', line)
                if match:
                    nsp = int(match.group(1))
                    params['species'] = [{'dtype': []} for i in range(nsp)]
                    continue

                # Set active species
                match = re.match(r'# species\((\d+)\)', line)
                if match:
                    active_species = int(match.group(1))-1
                elif line.startswith('# field data information'):
                    active_species = None

                # Read variable information
                match = re.match(r'"(.*)" (\w+) (\d+) (\w+) (\d+)', line)
                if match:
                    name = self.VARNAMES[match.group(1)]
                    dim = match.group(2)

                    if dim == 'scalar':
                        name = [name]
                    elif dim == 'vector':
                        name = [name + comp for comp in ['x', 'y', 'z']]
                    elif dim == 'tensor':
                        name = [name + comp for comp in ['xx', 'yy', 'zz',
                                                         'yz', 'zx', 'xy']]
                    else:
                        raise TypeError(f'Unknown data type "{dim}".')

                    dtype = match.group(4)[0] + match.group(5)
                    dtype = list(zip(name, [dtype]*len(name)))

                    if active_species is None:
                        params['field']['dtype'].extend(dtype)
                    else:
                        params['species'][active_species]['dtype'].extend(dtype)

        for fileinfo in [self._params['field']] + self._params['species']:
            fileinfo['dtype'] = np.dtype(fileinfo['dtype'])

    @property
    def datasets(self):
        """The available datasets."""
        datasets = []

        if 'data_base_filename' in self._params['field']:
            dsets = self._params['field']['dtype'].names
            dname = self._params['field']['data_directory']
            name = self._params['field']['data_base_filename']
            datasets.extend([f'{dname}/{name}/{dset}' for dset in dsets])

        for species in self._params['species']:
            dsets = species['dtype'].names
            dname = species['data_directory']
            name = species['data_base_filename']
            datasets.extend([f'{dname}/{name}/{dset}' for dset in dsets])

        return datasets

    def get_fileinfo(self, dataset):
        """Get descriptive information for the files containing `dataset`."""
        fileinfos = [self._params['field']] + self._params['species']
        for fileinfo in fileinfos:
            prefix = fileinfo['data_directory']+'/'+fileinfo['data_base_filename']
            if dataset.startswith(prefix+'/'):
                return fileinfo
        raise KeyError(f'Unknown dataset {dataset}.')

    def get_timesteps(self, dataset):
        """Get the available timesteps for `dataset`."""
        fileinfo = self.get_fileinfo(dataset)
        filedir = os.path.join(self._params['basedir'],
                               fileinfo['data_directory'])
        timesteps = []
        for filename in os.listdir(filedir):
            match = re.match(r'T.(\d+)', filename)
            if match:
                timesteps.append(int(match.group(1)))
        timesteps.sort()
        return np.array(timesteps)

    def get_local_grid(self, dataset, rank, timestep=0):
        """Get the 3D grid for a domain, defined on cell centers."""

        finfo = self.get_fileinfo(dataset)
        rankfile = os.path.join(self._params['basedir'],
                                finfo['data_directory'],
                                f'T.{timestep}',
                                f'{finfo["data_base_filename"]}.{timestep}.{rank}')

        mmap = np.memmap(rankfile, dtype='int32', shape=(16,), offset=31)

        # Load step and local grid
        file_step = mmap[0]
        file_rank = mmap[14]

        if file_step != timestep:
            raise IOError(f"Timestep mismatch in {rankfile}. "
                          f"Expected {timestep}, read {file_step}.")

        if file_rank != rank:
            raise IOError(f"Domain rank mismatch in {rankfile}. "
                          f"Expected {rank}, read {file_rank}.")

        f_nx = mmap[1:4]

        # Reinterpret as float
        mmap = mmap.view('float32')

        # Load spatial coordinates
        f_dx = mmap[5:8]
        f_x0 = mmap[8:11]
        f_x1 = f_x0 + (f_nx-1) * f_dx

        # Return the axes
        f_axes = [np.linspace(*x) for x in zip(f_x0, f_x1, f_nx)]

        if self._order == 'C':
            return f_axes[::-1]
        return f_axes

    def get_grid(self, dataset, merge_domains=True):
        """Get the 4D grid for the dataset, defined on cell centers."""

        # Find all the timesteps for this dataset
        tsteps = self.get_timesteps(dataset)
        if tsteps.size == 0:
            return [[0]]*4

        # Get rank 0 gird. This should always exist.
        axes = self.get_local_grid(dataset, 0, tsteps[0])
        topo = self._topology

        if not merge_domains:
            axes = [[x] for x in axes]

        # Now build in each F-ordered direction
        for axis in range(3):
            index = [0, 0, 0]

            # Build along a dimension
            for i in range(1, topo[axis]):
                index[axis] = i
                rank = np.ravel_multi_index(index, topo, order=self._order)
                rank_axes = self.get_local_grid(dataset, rank, tsteps[0])

                if merge_domains:
                    axes[axis] = np.hstack([axes[axis], rank_axes[axis]])
                else:
                    axes[axis].append(rank_axes[axis])

        # Add in timesteps
        t = tsteps*self._params['grid']['delta_t']

        if self._order == 'C':
            axes = [t] + axes
        else:
            axes = axes + [t]

        return axes

    def __getitem__(self, dataset):
        """Get a dataset."""

        # Get the datashape for individual files.
        if self._order == 'C':
            tsteps, *datashape = self.get_grid(dataset, merge_domains=False)
        else:
            *datashape, tsteps = self.get_grid(dataset, merge_domains=False)

        tsteps = np.round(tsteps/self._params['grid']['delta_t']).astype('int')

        for axis in range(3):
            datashape[axis] = np.asarray(list(map(len, datashape[axis])))

        # Load the info about the set of files containing dataset
        fileinfo = self.get_fileinfo(dataset)
        dataset = os.path.basename(dataset)
        basedir = self._params['basedir']
        filedir = os.path.join(basedir, fileinfo['data_directory'])
        prefix = fileinfo['data_base_filename']

        # Get the dtype, offset, and stride for a single file. Files can either
        # be banded or band-interleaved.
        dtype, offset = fileinfo['dtype'].fields[dataset]

        if self._interleave:

            # Offsets are constant across files.
            offset = offset + self._params['header_size']
            offset_func = lambda shape: offset

            # Stride is constant across files.
            stride = fileinfo['dtype'].itemsize
            assert stride % dtype.itemsize == 0
            stride = stride//dtype.itemsize

        else:

            # Add extra offset to account for ghosts.
            offset_func = lambda shape: offset*np.prod(shape) + self._params['header_size']

            # Stride is constant across files.
            stride = 1

        # Now contruct the array of files.
        datafiles = []
        for tstep in tsteps:
            stepdir = os.path.join(filedir, f'T.{tstep}')
            for rank in range(np.prod(self.topology)):
                datafiles.append(os.path.join(stepdir,
                                              f'{prefix}.{tstep}.{rank}'))

        # Correct shape for ordering
        topo = tuple(self.topology)
        nsteps = len(tsteps)
        if self._order == 'C':
            datashape = [np.ones(nsteps)] + datashape
            datafiles = np.array(datafiles).reshape((nsteps,) + topo, order='C')
            ghosts = (0, 1, 1, 1)
        else:
            datashape = datashape + [np.ones(nsteps)]
            datafiles = np.array(datafiles).reshape(topo + (nsteps,), order='F')
            ghosts = (1, 1, 1, 0)

        # Create the dataset.
        return MultiFileDataset(datafiles,
                                datashape,
                                dtype=dtype,
                                offset=offset_func,
                                stride=stride,
                                order=self._order,
                                ghosts=ghosts)
