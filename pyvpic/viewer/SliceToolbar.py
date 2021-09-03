import os
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class VPICToolbar(NavigationToolbar):

    # A new signal to show that we've sliced the data.
    sliced = QtCore.pyqtSignal(float, float, name='sliced')

    def __init__(self, *args, **kwargs):

        # Unpack.
        self.status_bar = kwargs.pop('status_bar', None)
        self.main_axes = kwargs.pop('main_axes', None)

        # Set up the base toolbar.
        kwargs['coordinates'] = False
        super().__init__(*args, **kwargs)

        # Now connect the status bar and the plot coordinate.
        if self.status_bar is not None:
            statusbar_label = QtWidgets.QLabel(parent=self.status_bar)
            self.status_bar.insertPermanentWidget(0, statusbar_label)
            self.message.connect(statusbar_label.setText)

        # Add new actions. There is actually another API for this using the
        # toolmanager, but currently there is a warning that it may change in
        # the future. So for now, we will just use this.

        # Add in the lineout tool.
        actions = self.actions()
        slice_icon = os.path.join(os.path.dirname(__file__), 'crosshairs.svg')
        slice_icon = QtGui.QIcon(slice_icon)
        slice_action = QtWidgets.QAction(slice_icon, 'Slice', self)
        slice_action.triggered.connect(self.slice)
        slice_action.setCheckable(True)
        slice_action.setToolTip("Set origin for slices")
        self.insertAction(actions[4], slice_action)
        self._actions['slice'] = slice_action


    def press_slice(self, event):
        """Handle a slice selection. Convert to axes coordinates and emit
        a signal."""
        if self.main_axes is not None and self.main_axes.in_axes(event):
            trans = self.main_axes.transData.inverted()
            self.sliced.emit(*trans.transform((event.x, event.y)))

    def _update_buttons_checked(self):
        """This is hackish, but works with current matplotlib.
        Overrides def in matplotlib.backends.backend_qt5.NavigationToolbar2QT"""
        super()._update_buttons_checked()

        if 'slice' in self._actions:
            self._actions['slice'].setChecked(self._active == 'SLICE')

    def _set_cursor(self, event):
        """Also hackish, but allows us to change the cursor.
        Overrides def in matplotlib.backend_bases.NavigationToolbar2"""
        active = self._active
        if active == 'SLICE':
            if self.main_axes.in_axes(event):
                self._active = 'ZOOM'
            else:
                self._active = False
        super()._set_cursor(event)
        self._active = active

    def slice(self, checked):
        """Handle enabling/disabling the slice button."""
        if self._active == 'SLICE':
            self._active = None
        else:
            self._active = 'SLICE'

        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''

        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''

        if self._active:
            self._idPress = self.canvas.mpl_connect('button_press_event',
                                                    self.press_slice)
            self.mode = 'slice origin'

        for axes in self.canvas.figure.get_axes():
            axes.set_navigate_mode(self._active)
        self._update_buttons_checked()
