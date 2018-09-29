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

    def get_grid(self, dataset):
        """Get the 4D grid for the dataset, defined on cell centers."""

        # Step from the header is not exactly right, due to floating point
        # issues and the possibility of strided output. To get a better
        # estimate, we will read in the header from the first rank on the
        # first timestep. This is not robust if user changes stride or has a
        # non-uniform decomposition, but these are extreme cases.
        finfo = self.get_fileinfo(dataset)
        tsteps = self.get_timesteps(dataset)

        if tsteps.size == 0:
            return [[0]]*4

        # Find the first file.
        testfile = os.path.join(self._params['basedir'],
                                finfo['data_directory'],
                                f'T.{tsteps[0]}',
                                f'{finfo["data_base_filename"]}.{tsteps[0]}.0')

        # Read in the header to find data dimensions in C-ordering.
        dims = np.memmap(testfile, dtype='int32', shape=(3,), offset=35)[::-1]

        if self._order == 'C':
            dims = dims * self.topology
        else:
            dims = dims * self.topology[::-1]

        # C-ordered grid
        grid = self._params['grid']
        axes = [tsteps*self._params['grid']['delta_t'],]
        for axis, axis_dim in zip(['z', 'y', 'x'], dims):
            start, stop = grid[f'extents_{axis}']
            axes.append((np.arange(axis_dim)+0.5)*(stop-start)/axis_dim + start)

        # F-ordered grid
        if self._order == 'F':
            axes = axes[::-1]

        return axes

    def __getitem__(self, dataset):
        """Get a dataset."""

        # Get the datashape for individual files.
        if self._order == 'C':
            tsteps, *datashape = self.get_grid(dataset)
        else:
            *datashape, tsteps = self.get_grid(dataset)
        tsteps = np.round(tsteps/self._params['grid']['delta_t']).astype('int')
        datashape = tuple(map(len, datashape))/self.topology

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
            offset = offset + self._params['header_size']
            stride = fileinfo['dtype'].itemsize
            assert stride % dtype.itemsize == 0
            stride = stride//dtype.itemsize
        else:
            # Add 2 to account for ghost cells.
            offset = offset*np.prod(datashape+2) + self._params['header_size']
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
        tsteps = (len(tsteps),)
        if self._order == 'C':
            datashape = (1,) + tuple(datashape)
            datafiles = np.array(datafiles).reshape(tsteps + topo, order='C')
            ghosts = (0, 1, 1, 1)
        else:
            datashape = tuple(datashape) + (1,)
            datafiles = np.array(datafiles).reshape(topo + tsteps, order='F')
            ghosts = (1, 1, 1, 0)

        # Create the dataset.
        return MultiFileDataset(datafiles,
                                datashape,
                                dtype=dtype,
                                offset=offset,
                                stride=stride,
                                order=self._order,
                                ghosts=ghosts)
