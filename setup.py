from setuptools import setup, find_packages

setup(
    name     = 'pyvpic',
    author   = 'VPIC Team',
    url      = '',
    version  = '0.1dev',
    packages = find_packages(),
    python_requires  = '>=3.6',
    install_requires = [
        'numpy',
        'matplotlib',
        'h5py',
    ],
    entry_points = {
        'gui_scripts': ['vpicviewer=pyvpic.viewer:_main_command_line'],
    },
    include_package_data = True
)
