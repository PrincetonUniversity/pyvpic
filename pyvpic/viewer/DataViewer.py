import os.path
import logging
import numpy as np
from PyQt5 import QtCore, QtWidgets, uic
import pyvpic.data as vpicdata
from pyvpic.readers.BaseReader import BaseReader
from .TreeModel import TreeModel
from .Canvas import VPICCanvas
from .SliceToolbar import VPICToolbar
from .Logger import log_timing

# Much of the UI design including signals and slots is done in QT Creator. The
# result of that is the viewer.ui file that we are loading here. All we need to
# do is add the logic.
UIFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "viewer.ui"))

# The main widget with added logic.
class VPICDataViewer(QtWidgets.QMainWindow, uic.loadUiType(UIFILE)[0]):

    AXES = {'Time': 0, 'Z': 1, 'Y': 2, 'X': 3}

    def __init__(self, fpath, statusbar_label=None, **kwargs):
        super().__init__()

        # Setup the base UI from the design template.
        self.setupUi(self)

        # Add in the matplotlib figure widget.
        layout = QtWidgets.QVBoxLayout(self.PlotFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        self.fig = VPICCanvas(self.PlotFrame)
        layout.addWidget(self.fig)

        # Add the toolbar
        nav = VPICToolbar(self.fig, self.PlotFrame,
                          status_bar=self.statusBar,
                          main_axes=self.fig.axes[0, 1])
        nav.sliced.connect(self.set_slice)
        self.addToolBar(nav)

        # Make things tight.
        self.verticalLayout_3.setContentsMargins(7, 0, 7, 0)

        # Defaults.
        if fpath is None:
            self.vpic_io = BaseReader()
        else:
            self.vpic_io = vpicdata.open(fpath, **kwargs)
        self.dataset = np.empty([0, 0, 0, 0])
        self.kwargs = kwargs

        # Set up the tree widget.
        self.tree_model = TreeModel(self.vpic_io.tree)
        self.TreeView.setModel(self.tree_model)

        # Connect the logger.
        if statusbar_label is not None:
            self.statusBar.addWidget(statusbar_label, stretch=1)

        # Start the logging.
        self.logger = logging.getLogger('vpic.viewer')
        self.logger.info('Loading dataset: empty')

        # Initialize the axes
        self.update_axes()

    def file_dialog(self):
        """Slot to change the data source.
        Called by actionOpenFile on signal 'triggered'"""
        # Select a file using a dialog.
        options = QtWidgets.QFileDialog.Options()
        filters = ['VPIC Files (*.vpc)']
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Data', '',
                                                         ';;'.join(filters),
                                                         options=options)
        if fname != '':
            self.change_source(fname)

    def dir_dialog(self):
        """Slot to change the data source.
        Called by actionOpenDir on signal 'triggered'"""
        # Select a directory using a dialog.
        options = QtWidgets.QFileDialog.Options()
        dname = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Folder',
                                                           '', options=options)
        if dname != '':
            self.change_source(dname)

    def change_source(self, fname):
        """Open a reader."""
        try:
            self.vpic_io = vpicdata.open(fname, **self.kwargs)
        except IOError as ioerr:
            self.logger.error(str(ioerr))
            return
        self.tree_model.setTree(self.vpic_io.tree)
        self.change_dataset()

    def change_dataset(self):
        """Slot to change the dataset.
        Called by TreeView on signal 'clicked'"""
        dataset = self.tree_model.data(self.TreeView.currentIndex(),
                                       QtCore.Qt.UserRole)

        if dataset is not None:
            self.logger.info(f'Loading dataset: {dataset}')
            self.load_metadata(dataset)
            self.fig.read_data()
            self.fig.draw_plots(rescale=True)

    @log_timing
    def load_metadata(self, dataset):
        """Load the metadata for a dataset."""
        # Load the dataset metadata
        self.dataset = self.vpic_io[dataset]

        # Update axes and figure properties
        self.update_axes(update=False)
        self.fig.dataset = self.dataset
        self.fig.title = dataset
        self.fig.grid = self.vpic_io.get_grid(dataset)

    def update_axes(self, update=True):
        """Slot to set the axes dropdowns.
        Called by XAxis on signal 'activated'
        Called by YAxis on signal 'activated'
        """
        # See if we need to update.
        choices = ['X', 'Y', 'Z', 'Time']

        # Remove axes one at a time.
        self.set_combo(self.XAxis, choices)
        self.set_combo(self.YAxis, choices)
        self.set_slider(self.Z1Slider, self.Z1Label, choices.pop(0))
        self.set_slider(self.Z2Slider, self.Z2Label, choices.pop(0))

        xaxis = self.AXES[self.XAxis.currentText()]
        yaxis = self.AXES[self.YAxis.currentText()]

        if xaxis != self.fig.xaxis or yaxis != self.fig.yaxis:
            self.fig.xaxis = xaxis
            self.fig.yaxis = yaxis
            if update:
                self.fig.read_data()
                self.fig.draw_plots(rescale=True)

    def update_sliders(self):
        """Slot to set the out-of-plane slices.
        Called by Z1Slider on signal 'sliderReleased'
        Called by Z2Slider on signal 'sliderReleased'
        """
        text = self.Z1Label.text()
        self.fig.set_slice(self.AXES[text], self.Z1Slider.sliderPosition())

        text = self.Z2Label.text()
        self.fig.set_slice(self.AXES[text], self.Z2Slider.sliderPosition())

        self.fig.read_data()
        self.fig.draw_plots()

    def set_combo(self, combo, axes):
        """Update a combo box label and choices."""
        text = combo.currentText()
        combo.clear()
        combo.insertItems(0, axes)
        if text in axes:
            combo.setCurrentText(text)
        axes.remove(combo.currentText())

    def set_slider(self, slider, label, text):
        """Update the slider labels and bounds."""

        if self.dataset.size > 0:
            shape = self.dataset.shape[self.AXES[text]]
        else:
            shape = 1

        label.setText(text)
        slider.setMinimum(0)
        slider.setMaximum(shape-1)

        # Set the slice.
        self.fig.set_slice(self.AXES[text], slider.sliderPosition())

    def set_slice(self, xslice, yslice):
        """Slot to set the in-plane slices from a click on the figure.
        Called by VPICToolbar on signal 'sliced'
        """
        self.fig.set_slice(self.AXES[self.XAxis.currentText()], float(xslice))
        self.fig.set_slice(self.AXES[self.YAxis.currentText()], float(yslice))
        self.fig.draw_plots()
