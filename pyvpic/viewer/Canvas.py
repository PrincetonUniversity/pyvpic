import logging
import numpy as np
from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .Logger import log_timing
__all__ = ('VPICCanvas',)

class VPICCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):

        # Logging
        self.logger = logging.getLogger('vpic.canvas')

        # Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(2, 3,
                                      gridspec_kw={'width_ratios': [1, 16, 4],
                                                   'height_ratios': [4, 1],
                                                   'wspace': 0.05,
                                                   'hspace': 0.05,
                                                   'left': 0.12,
                                                   'right': 0.9,
                                                   'bottom': 0.1,
                                                   'top': 0.93})

        # Delete un-needed axes.
        self.axes[1, 0].remove()
        self.axes[1, 2].remove()

        # Share axes.
        self.axes[0, 1].get_shared_x_axes().join(self.axes[0, 1],
                                                 self.axes[1, 1])
        self.axes[0, 1].get_shared_y_axes().join(self.axes[0, 1],
                                                 self.axes[0, 2])

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.dataset = np.empty([0, 0, 0, 0])
        self.title = 'null'
        self.grid = [np.zeros(1)]*4
        self.slicer = [slice(0, 1)]*4
        self.xaxis = 0
        self.yaxis = 1
        self.vline = None
        self.hline = None
        self.image = None
        self.data = np.empty([0, 0, 0, 0])
        self.data_stale = True
        self.cbar = None

    @log_timing
    def read_data(self):
        """Read and cache the 2D slice. By caching, we speed up lineouts."""
        if self.dataset.size > 0:
            slicer_xy = self.slicer.copy()
            slicer_xy[self.xaxis] = slice(None)
            slicer_xy[self.yaxis] = slice(None)
            self.data = self.dataset[tuple(slicer_xy)]
        else:
            self.data = np.empty([0, 0, 0, 0])
        self.data_stale = True

    def set_slice(self, axis, index):
        """Set the slice indicies for the line-outs."""
        # Data coordinate.
        if isinstance(index, float):
            index = np.searchsorted(self.grid[axis], index)
            index = max(0, index-1)

        # Exact index.
        self.slicer[axis] = slice(index, index+1)

    def clear(self):
        """Clear existing plots."""
        self.axes[0, 0].cla()
        self.axes[0, 1].cla()
        self.axes[1, 1].cla()
        self.axes[0, 2].cla()

    def draw_2d(self, extent):
        """Main 2D plotting. We only need to redraw this if the underlying
        data has changed."""

        # Grab the data.
        squeeze_axis = [0, 1, 2, 3]
        squeeze_axis.remove(self.xaxis)
        squeeze_axis.remove(self.yaxis)
        if self.data.size > 0:
            data_xy = np.squeeze(self.data, axis=tuple(squeeze_axis))
            if self.xaxis < self.yaxis:
                data_xy = data_xy.T
        else:
            data_xy = np.zeros((1, 1))

        if self.image is not None:
            # Just update the underlying data. We do this because if users
            # update preferences (colormap, interpolation, etc.) through the
            # GUI it will be perserved across updates.
            self.image.set_data(data_xy)
            self.image.set_extent(extent)
            self.image.autoscale()

        else:
            # Draw the initial plot.
            self.image = self.axes[0, 1].imshow(data_xy,
                                                origin='lower',
                                                aspect='auto',
                                                extent=extent)

            # Colobar construction
            self.cbar = self.fig.colorbar(self.image, cax=self.axes[0, 0])

        if self.cbar is not None:
            self.cbar.ax.tick_params(axis='y',
                                     right=False, labelright=False,
                                     left=True, labelleft=True)

        # Set tickparams.
        self.axes[0, 1].tick_params(axis='both',
                                    right=False, labelright=False,
                                    left=False, labelleft=False,
                                    bottom=False, labelbottom=False,
                                    top=False, labeltop=False)

    @log_timing
    def draw_plots(self, rescale=False):
        """Draw plots for the dataset and slices."""
        if self.dataset.size > 0:

            xgrid = self.grid[self.xaxis]
            ygrid = self.grid[self.yaxis]
            xslice = xgrid[self.slicer[self.xaxis]][0]
            yslice = ygrid[self.slicer[self.yaxis]][0]

            slicer_x = [slice(None)]*4
            slicer_x[self.yaxis] = self.slicer[self.yaxis]

            slicer_y = [slice(None)]*4
            slicer_y[self.xaxis] = self.slicer[self.xaxis]

            # Assumes these should be 1D, but is better than squeeze due
            # to implementation inconsistency in squeezing memmaps.
            data_x = self.data[tuple(slicer_x)].flatten()
            data_y = self.data[tuple(slicer_y)].flatten()

        else:
            data_x = np.zeros(2)
            data_y = np.zeros(2)
            xgrid = np.atleast_1d([0, 1])
            ygrid = np.atleast_1d([0, 1])
            xslice = 0
            yslice = 0


        # Save the current limits.
        if rescale:
            xlim = xgrid[[0, -1]]
            ylim = ygrid[[0, -1]]
        else:
            xlim = self.axes[0, 1].get_xlim()
            ylim = self.axes[0, 1].get_ylim()

        # If stale, redraw the image.
        if self.data_stale:
            self.draw_2d([xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]])
            self.data_stale = False

        # Construct or move the cursors.
        if self.vline is None:
            self.vline = self.axes[0, 1].axvline(xslice, c='0.2')
        else:
            self.vline.set_data([xslice, xslice], [0, 1])

        if self.hline is None:
            self.hline = self.axes[0, 1].axhline(yslice, c='0.2')
        else:
            self.hline.set_data([0, 1], [yslice, yslice])

        # Plot the lineouts.
        self.axes[0, 2].cla()
        self.axes[1, 1].cla()
        self.axes[0, 2].plot(data_y, ygrid, c='0.2', scaley=False)
        self.axes[1, 1].plot(xgrid, data_x, c='0.2', scalex=False)

        # Set limits
        self.axes[0, 1].set_xlim(*xlim)
        self.axes[0, 1].set_ylim(*ylim)

        # Set labels
        labels = ['t', 'z', 'y', 'x']
        self.axes[0, 2].set_ylabel(rf'${labels[self.yaxis]}$', fontsize='x-large')
        self.axes[0, 2].yaxis.set_label_position('right')
        self.axes[0, 2].yaxis.label.set_rotation(0)
        self.axes[0, 2].yaxis.labelpad = 10
        self.axes[1, 1].set_xlabel(rf'${labels[self.xaxis]}$', fontsize='x-large')

        # Set title
        indices = list(range(4))
        indices.remove(self.xaxis)
        indices.remove(self.yaxis)

        slice_labels = [rf'${labels[i]} = {self.grid[i][self.slicer[i]][0]:.4g}$'
                        for i in indices]
        title = self.title + ': ' + ', '.join(slice_labels)
        self.axes[0, 1].set_title(title, size='large')

        # Set tickparams
        self.axes[0, 2].tick_params(axis='both',
                                    right=True, labelright=True,
                                    left=False, labelleft=False,
                                    bottom=False, labelbottom=False,
                                    top=True, labeltop=True)

        self.axes[1, 1].tick_params(axis='both',
                                    right=False, labelright=False,
                                    left=True, labelleft=True,
                                    bottom=True, labelbottom=True,
                                    top=False, labeltop=False)

        self.draw()
