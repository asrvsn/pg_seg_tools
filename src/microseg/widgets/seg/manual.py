'''
Manual segmentor dialog
'''
from qtpy.QtWidgets import QRadioButton, QCheckBox, QButtonGroup

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *

class ROICreatorWidget(VLayoutWidget):
    '''
    Create ROIs from polygons
    '''
    edited = QtCore.Signal(object) # List[ROI]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._poly_wdg = HLayoutWidget()
        self._poly_btn = QRadioButton('Polygon')
        self._poly_wdg.addWidget(self._poly_btn)
        self._chull_box = QCheckBox('Convex hull')
        self._poly_wdg.addWidget(self._chull_box)
        self.addWidget(self._poly_wdg)
        self._ellipse_btn = QRadioButton('Ellipse')
        self.addWidget(self._ellipse_btn)
        self._circle_btn = QRadioButton('Circle')
        self.addWidget(self._circle_btn)
        self._roi_grp = QButtonGroup(self)
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn]:
            self._roi_grp.addButton(btn)

        # State
        self._polys = []
        self._poly_btn.setChecked(True)
        self._chull_box.setChecked(False)

        # Listeners
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn, self._chull_box]:
            btn.toggled.connect(self._recompute)

    def setData(self, polys: List[PlanarPolygon]):
        self._polys = polys
        self._recompute()

    def _recompute(self):
        mk_poly = self._poly_btn.isChecked()
        if mk_poly:
            self._chull_box.setEnabled(True)
        else:
            self._chull_box.setEnabled(False)
        use_chull = self._chull_box.isChecked()
        mk_ell = self._ellipse_btn.isChecked()
        mk_circ = self._circle_btn.isChecked()
        rois = []
        for poly in self._polys:
            if mk_poly:
                if use_chull: 
                    roi = poly.hullify()
                else:
                    roi = poly
            elif mk_ell:
                roi = Ellipse.from_poly(poly)
            elif mk_circ:
                roi = Circle.from_poly(poly)
            else:
                raise Exception('Invalid ROI type')
            rois.append(roi)
        self.edited.emit(rois)

class ManualSegmentorWidget(SegmentorWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._creator = ROICreatorWidget()
        self._main.addWidget(self._creator)
        self._creator.edited.connect(self.propose.emit) # Bubble from the editor

    def name(self) -> str:
        return 'Manual'

    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[ROI]:
        self._creator.setData([poly])