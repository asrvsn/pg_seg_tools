'''
Manual segmentor dialog
'''
from qtpy.QtWidgets import QRadioButton, QCheckBox

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *

class ManualSegmentorWidget(SegmentorWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._poly_wdg = HLayoutWidget()
        self._poly_btn = QRadioButton('Polygon')
        self._poly_wdg.addWidget(self._poly_btn)
        self._chull_box = QCheckBox('Convex hull')
        self._poly_wdg.addWidget(self._chull_box)
        self._main.addWidget(self._poly_wdg)
        self._ellipse_btn = QRadioButton('Ellipse')
        self._main.addWidget(self._ellipse_btn)
        self._circle_btn = QRadioButton('Circle')
        self._main.addWidget(self._circle_btn)

        # State
        self._poly_btn.setChecked(True)
        self._chull_box.setChecked(True)

        # Listeners
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn, self._chull_box]:
            btn.toggled.connect(self._recompute)

    ''' Overrides '''

    def name(self) -> str:
        return 'Manual'

    def make_proposals(self, poly: PlanarPolygon) -> List[ROI]:
        if self._poly_btn.isChecked():
            if self._chull_box.isChecked():
                poly = poly.hullify()
            return [poly]
        elif self._ellipse_btn.isChecked():
            return [Ellipse.from_poly(poly)]
        elif self._circle_btn.isChecked():
            return [Circle.from_poly(poly)]
        else:
            raise NotImplementedError
        
    ''' Private methods '''

    def _recompute(self):
        if self._poly_btn.isChecked():
            self._chull_box.setEnabled(True)
        else:
            self._chull_box.setEnabled(False)
        self._propose()