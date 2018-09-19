# pyvpic - Python tools for VPIC data analysis

## Installation
`pyvpic` is designed to work with `setuptools` so you can install using
`pip`. For a local install:
```
pip install --user .
```
## Usage

### Data Access
Readers for multiple VPIC file formats are provided. These readers can be used directly, or the `pyvpic.open` function can be used to automatically choose the appropriate reader. Readers provide a simplified memory mapped view into the actual data files. Each dataset is a 4D array. With the default C-ordering, the datasets are indexed as `(time, z, y, x)`. All readers also support an option Fortran ordering by setting `order='F'` in which case datasets are indexed as `(x, y, z, time)`.

#### Examples
```python
>>> import pyvpic
>>> import matplotlib.pyplot as plt
>>> reader = pyvpic.open('global.vpc', order='C')
>>> reader.datasets
['fields/bx',
 'fields/by',
 'fields/bz',
...]
>>> bz = reader['fields/bz']
>>> bz.shape
(128, 260, 1, 130)
>>> t, z, y, x = reader.get_grid('fields/bz')
>>> plt.pcolormesh(z, x, bz[0,:,0,:].T)
```

### Data Visualization
A simple data visualization tool is provided for quickly viewing 1D and 2D slices of VPIC data. If installed via `pip`, this viewer is available as a command line program. Command line arguments are the same as those passed to `pyvpic.open`, however `order` is always ignored if present.

```bash
# Launch the viewer to read interleaved file-per-rank data.
$ vpicviewer global.vpc --interleave=True
```

The viewer can also be launched from within an interactive python session.

```python
>>> import pyvpic.viewer
>>> pyvpic.viewer.run('global.vpc', interleave=True)
```


### Data File Specifications

#### File Per Rank
The `pyvpic.readers.FilePerRankReader` reader maps the individual datafiles output by VPIC into a 4D dataset. This is done lazily and no data is loaded until the dataset is actually sliced, at which point only files containing the requested data are accessed. To use the File per rank reader, pass a VPIC header file, `global.vpc` to `pyvpic.open`. Additionally, VPIC has the option of either outputting banded or band-interleaved data. This choice is currently not recorded in the VPIC header file and must be passed to the reader as well.

#### GDA Files
GDA files are straight, brick-of-value datafiles with no padding or headers. Data is assumed to be stored using `'float32'` format and C-ordered. GDA comes in two flavors both of which are supported by `pyvpic.readers.GDAReader`:

1. **Single-file, multiple-timesteps** - In this case all data is contained within a single file and the 4D dataset maps directly to the file. The number of timesteps is determined from the length of the file.
2. **Single-file, single-timestep** - In this case, each timestep is stored as a single file with the naming convention `<dataset>_<timestep>.gda`. Different files are transparently mapped to different slices of the 4D dataset.

To use the `pyvpic.readers.GDAReader`, a directory containing GDA files should be passed. By default, the directory is scanned recursively for all GDA files. Each directory containing GDA files should also contain a binary `info` file structured as:

```
<Start of file: info>
--      - padding              - 4 bytes
nx      - number of cells in x - float32
ny      - number of cells in y - float32
nz      - number of cells in z - float32
--      - padding              - 8 bytes
Lx      - Domain length in x   - float32
Ly      - Domain length in y   - float32
Lz      - Domain length in z   - float32
...
```


