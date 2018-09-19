import numpy as np

class BaseReader:
    """A base class defining the generic interface for VPIC data. All readers
    should inherit from this class, but it should not be instantiated directly.

    Methods
    -------


    """

    def __init__(self, **kwargs):
        pass

    def get_grid(self, dataset):
        return [np.empty(0)]*4

    def __getitem__(self, dataset):
        return np.empty((0, 0, 0, 0))

    @property
    def datasets(self):
        """A list of the available datasets."""
        return []

    @property
    def tree(self):
        """A hierarchical representation of the available datasets."""
        tree = {'groups': {}, 'datasets': []}
        datasets = list(self.datasets)
        datasets.sort()
        for dset in datasets:
            group = tree
            *groups, name = dset.split('/')
            while groups:
                next_group = groups.pop(0)
                if next_group not in group['groups']:
                    group['groups'][next_group] = {'groups': {}, 'datasets': []}
                group = group['groups'][next_group]
            group['datasets'].append(name)

        return tree
