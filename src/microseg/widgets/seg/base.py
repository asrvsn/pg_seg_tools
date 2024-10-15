'''
Base classes for turning prompt user polygons into segmentations
Segmentor widgets are basically floating menus
'''
import abc
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from matgeo import PlanarPolygon
from microseg.widgets.base import *
from microseg.widgets.roi import ROI

class SegmentorWidget(QtWidgets.QWidget, metaclass=abc.ABCMeta):
    propose = QtCore.Signal(object) # List[ROI]
    add = QtCore.Signal(object) # List[ROI]
    cancel = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        ''' Create self '''
        super().__init__(*args, **kwargs)
        self._layout = VLayout()
        self.setLayout(self._layout)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle(self.name())

    # Abstract static method: name
    @abc.abstractmethod
    @staticmethod
    def name() -> str:
        pass

    @abc.abstractmethod
    def prompt(self, poly: PlanarPolygon, show: bool):
        '''
        From a prompt polygon, produce a list of candidate ROIs
        the "show" parameter determines whether to show the options, with sensible 
        defaults taken if it is false. 
        - show=False: fire propose() any number of times then add() or cancel()
        - show=True: fire add() immediately
        '''
        pass