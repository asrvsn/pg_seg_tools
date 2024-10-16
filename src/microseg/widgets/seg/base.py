'''
Base classes for turning prompt user polygons into segmentations
Segmentor widgets are basically floating menus
'''
import abc
import numpy as np
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton
from qtpy.QtCore import Qt

from matgeo import PlanarPolygon
from microseg.widgets.base import *
from microseg.widgets.roi import ROI

class SegmentorWidget(VLayoutWidget, metaclass=QtABCMeta):
    propose = QtCore.Signal(object) # List[ROI]
    add = QtCore.Signal() 
    cancel = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._main = VLayoutWidget()
        self.addWidget(self._main)
        self._bottom = HLayoutWidget()
        self._ok_btn = QPushButton('OK')
        self._bottom.addWidget(self._ok_btn)
        self._cancel_btn = QPushButton('Cancel')
        self._bottom.addWidget(self._cancel_btn)
        self.addWidget(self._bottom)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle(self.name())

        # State
        self.reset_state()

        # Listeners
        self._ok_btn.clicked.connect(self._ok)
        self._cancel_btn.clicked.connect(self._cancel)
        
    ''' Overrides '''

    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        From a prompt image and polygon, produce a list of candidate ROIs, given the current settings.
        Should fire the propose() signal at least one synchronously using these ROIs.
        '''
        pass

    def reset_state(self):
        '''
        Do any state resets in here before the next call.
        '''
        self._img = None
        self._poly = None

    ''' API '''

    def prompt(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Lets the user to fire propose() or cancel() using buttons.
        '''
        self._img = img
        self._poly = poly
        self.show()
        self.make_proposals(self._img, self._poly)

    def prompt_immediate(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Fires the add() signal immediately.
        '''
        self.make_proposals(img, poly)
        self.add.emit()
        self.reset_state()

    ''' Private methods '''

    def _ok(self):
        assert not self._poly is None
        self.hide()
        self.add.emit()
        self.reset_state()

    def _cancel(self):
        self.hide()
        self.cancel.emit()
        self.reset_state()