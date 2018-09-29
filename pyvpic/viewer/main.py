import logging
import argparse
from PyQt5 import QtWidgets
from .DataViewer import VPICDataViewer
from .Logger import QLogLabel

def _main_command_line():
    """ Entry point to launch the viewer program from the command line.

    Command line arguments are parsed using argparse. To make this work nicely,
    options for each type of reader should be reproduced here.
    """
    parser = argparse.ArgumentParser(description='View VPIC data')
    parser.add_argument('fpath',
                        type=str,
                        nargs='?',
                        default='',
                        help='Data source location')

    # Options for FilePerRankReader
    rank_group = parser.add_argument_group('File Per Rank Reader',
                                           'Options for using the File Per '
                                           'Rank Reader to read raw VPIC '
                                           'output. This is used when '
                                           '`fpath` points to a .vpc file.')
    rank_group.add_argument('-i', '--interleave',
                            dest='interleave',
                            action='store_true',
                            help='Use band-interleaved format (default).')
    rank_group.add_argument('-b', '--banded',
                            dest='interleave',
                            action='store_false',
                            help='Use banded format.')
    rank_group.set_defaults(interleave=True)

    # Options for GDAReader
    gda_group = parser.add_argument_group('GDA Reader',
                                          'Options for using the GDA '
                                          'Reader. This is used when '
                                          '`fpath` points to a directory.')
    gda_group.add_argument('-r', '--recursive',
                           dest='recursive',
                           action='store_true',
                           help='Scan directories recursively (default).')
    gda_group.add_argument('-n', '--norecurse',
                           dest='recursive',
                           action='store_false',
                           help='Do not scan recursively.')
    gda_group.set_defaults(recursive=True)

    # Pass arguments on to the program.
    kwargs = vars(parser.parse_args())
    if kwargs['fpath'] == '':
        kwargs['fpath'] = None

    main(**kwargs)

def main(fpath=None, **kwargs):
    """ Launches the viewer program from a script or interactive session.

    Parameters
    ----------
    fpath: path, optional
        The file path to open on launch. Will be passed to `pyvpic.open`.
    **kwargs:
        Additional keyword arguments are passed to `pyvpic.open`.
    """
    # Start QT
    app = QtWidgets.QApplication([])

    # Set up logging.
    log = logging.getLogger('vpic')
    log.setLevel(logging.INFO)

    # Direct output to console.
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('{name:11} : {message}', style='{')
    console.setFormatter(formatter)
    log.addHandler(console)

    # Direct output to statusbar.
    status = QLogLabel()
    log.addHandler(status)

    # Now run the program.
    log.info('Launching viewer')
    window = VPICDataViewer(fpath, statusbar_label=status, **kwargs)
    window.show()
    app.exec_()

    # Manually remove QLogLabel to avoid error on exit.
    log.removeHandler(status)
    return
