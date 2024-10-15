'''
Base classes for turning prompt user polygons into segmentations
Segmentor widgets are basically floating menus
'''
import abc
from qtpy import QtWidgets
from qtpy.QtWidgets import QPushButton
from qtpy.QtCore import Qt

from matgeo import PlanarPolygon
from microseg.widgets.base import *
from microseg.widgets.roi import ROI

class SegmentorWidget(QtWidgets.QWidget, metaclass=abc.ABCMeta):
    propose = QtCore.Signal(object) # List[ROI]
    add = QtCore.Signal(object) # List[ROI]
    cancel = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._layout = VLayout()
        self.setLayout(self._layout)
        self._main = VLayoutWidget()
        self._layout.addWidget(self._main)
        self._bottom = HLayoutWidget()
        self._propose_btn = QPushButton('Propose')
        self._bottom.addWidget(self._propose_btn)
        self._ok_btn = QPushButton('OK')
        self._bottom.addWidget(self._ok_btn)
        self._cancel_btn = QPushButton('Cancel')
        self._bottom.addWidget(self._cancel_btn)
        self._layout.addWidget(self._bottom)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle(self.name())

        # State
        self._poly = None

        # Listeners
        self._propose_btn.clicked.connect(self._propose)
        self._ok_btn.clicked.connect(self._ok)
        self._cancel_btn.clicked.connect(self._cancel)
        
    ''' Overrides '''

    @abc.abstractmethod
    @staticmethod
    def name() -> str:
        pass

    @abc.abstractmethod
    def make_proposals(self, poly: PlanarPolygon) -> List[ROI]:
        '''
        From a prompt polygon, produce a list of candidate ROIs, given the current settings.
        '''
        pass

    ''' API '''

    def prompt(self, poly: PlanarPolygon):
        '''
        Lets the user to fire propose() or cancel() using buttons.
        '''
        self._poly = poly
        self.show()

    def prompt_immediate(self, poly: PlanarPolygon):
        '''
        Fires the add() signal immediately.
        '''
        self.add.emit(self.make_proposals(poly))

    ''' Private methods '''

    def _propose(self):
        assert not self._poly is None
        self.propose.emit(self.make_proposals(self._poly))

    def _ok(self):
        assert not self._poly is None
        self.hide()
        self.add.emit(self.make_proposals(self._poly))
        self._poly = None

    def _cancel(self):
        self.hide()
        self.cancel.emit()
        self._poly = None