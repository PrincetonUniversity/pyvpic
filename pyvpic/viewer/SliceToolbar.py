import os
import enum
from pkg_resources import packaging as pkg
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib


MPL_VERSION = pkg.version.parse(matplotlib.__version__)


class Mode(str, enum.Enum):
    NONE = ""
    PAN = "pan/zoom"
    ZOOM = "zoom rect"
    SLICE = "slice origin"

    def __str__(self):
        return self.value

    @property
    def _navigate_mode(self):
        return self.name if self is not _Mode.NONE else None


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

    def _update_buttons_checked_mpl31x(self):
        """This is hackish, but works with current matplotlib.
        Overrides def in matplotlib.backends.backend_qt5.NavigationToolbar2QT.
        Works in matplotlib <= 3.1.x"""
        super()._update_buttons_checked()
        self._actions['slice'].setChecked(self._active == 'SLICE')

    def _update_buttons_checked_mpl32x(self):
        """This is hackish, but works with current matplotlib.
        Overrides def in matplotlib.backends.backend_qt5.NavigationToolbar2QT.
        Works in matplotlib == 3.2.x"""
        super()._update_buttons_checked()

        if 'slice' in self._actions:
            self._actions['slice'].setChecked(self._active == 'SLICE')

    def _update_buttons_checked_mpl33x(self):
        """This is hackish, but works with current matplotlib.
        Overrides def in matplotlib.backends.backend_qt5.NavigationToolbar2QT.
        Works in matplotlib >= 3.3.x"""
        super()._update_buttons_checked()

        if 'slice' in self._actions:
            self._actions['slice'].setChecked(self.mode.name == 'SLICE')

    def _update_buttons_checked(self):
        """Update checked buttons based on matplotlib version."""
        if MPL_VERSION < pkg.version.parse('3.2.0'):
            self._update_buttons_checked_mpl31x()
        elif MPL_VERSION < pkg.version.parse('3.3.0'):
            self._update_buttons_checked_mpl32x()
        else:
            self._update_buttons_checked_mpl33x()

    def _set_cursor(self, event):
        """Also hackish, but allows us to change the cursor.
        Overrides def in matplotlib.backend_bases.NavigationToolbar2.
        Works with matplotlib <= 3.1.x"""
        active = self._active
        if active == 'SLICE':
            if self.main_axes.in_axes(event):
                self._active = 'ZOOM'
            else:
                self._active = False
        super()._set_cursor(event)
        self._active = active

    def _update_cursor_mpl32x(self, event):
        """Also hackish, but allows us to change the cursor.
        Overrides def in matplotlib.backend_bases.NavigationToolbar2.
        Works with matplotlib == 3.2.x"""
        active = self._active
        if active == 'SLICE':
            if self.main_axes.in_axes(event):
                self._active = 'ZOOM'
            else:
                self._active = False
        super()._update_cursor(event)
        self._active = active

    def _update_cursor_mpl33x(self, event):
        """Also hackish, but allows us to change the cursor.
        Overrides def in matplotlib.backend_bases.NavigationToolbar2.
        Works with matplotlib >= 3.3.x"""
        mode = self.mode
        if mode == Mode.SLICE:
            if self.main_axes.in_axes(event):
                self.mode = Mode.ZOOM
            else:
                self.mode = Mode.NONE
        super()._update_cursor(event)
        self.mode = mode

    def _update_cursor(self, event):
        """Update cursor based on matplotlib version."""
        if MPL_VERSION < pkg.version.parse('3.3.0'):
            self._update_cursor_mpl32x(event)
        else:
            self._update_cursor_mpl33x(event)

    def slice_mpl32x(self, *args):
        """Handle enabling/disabling the slice button. Works in
        matplotlib < 3.3.x"""
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

    def slice_mpl33x(self, *args):
        """Handle enabling/disabling the slice button. Works in
        matplotlib >= 3.3.x"""
        if self.mode == Mode.SLICE:
            self.mode = Mode.NONE
            self.canvas.widgetlock.release(self)
        else:
            self.mode = Mode.SLICE
            self.canvas.widgetlock(self)

        for axes in self.canvas.figure.get_axes():
            axes.set_navigate_mode(self.mode)

        self.set_message(self.mode)
        self._update_buttons_checked()

    def slice(self, *args):
        """Handle enabling/disabling the slice button."""
        if MPL_VERSION < pkg.version.parse('3.3.0'):
            self.slice_mpl32x(*args)
        else:
            self.slice_mpl33x(*args)

    def _zoom_pan_handler(self, event):
        """Hackish, but handles mouse events in matplotlib >= 3.3.0."""
        super()._zoom_pan_handler(event)
        if self.mode == Mode.SLICE:
            if event.name == 'button_press_event':
                self.press_slice(event)
