'''
Manual segmentor dialog
'''
from qtpy.QtWidgets import QRadioButton, QCheckBox, QButtonGroup

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *

class TouchpadWidget(QWidget):
    moved = QtCore.Signal(object) # np.array (x, y) offset
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self._mousedown_pos = None

    def mousePressEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self._mousedown_pos = evt.pos()

    def mouseMoveEvent(self, evt):
        if not self._mousedown_pos is None:
            offset = np.array([
                evt.pos().x() - self._mousedown_pos.x(),
                evt.pos().y() - self._mousedown_pos.y()
            ])
            self.moved.emit(offset)
            self._mousedown_pos = evt.pos()

    def mouseReleaseEvent(self, evt):
        if evt.button() == Qt.LeftButton:
            self._mousedown_pos = None


class ROICreatorWidget(VLayoutWidget):
    '''
    Create ROIs from polygons
    '''
    edited = QtCore.Signal(object) # List[ROI]
    MOVE_SCALE = 0.3
    
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
        self.addSpacing(10)
        self._scale_sld = FloatSlider(step=0.01)
        scale_wdg = HLayoutWidget()
        scale_wdg.addWidget(QLabel('Scale:'))
        scale_wdg.addWidget(self._scale_sld)
        self.addWidget(scale_wdg)
        touch_grp = VGroupBox('Move')
        self._touchpad = TouchpadWidget()
        self._touchpad.setFixedSize(200, 133)
        touch_grp.addWidget(self._touchpad)
        self.addWidget(touch_grp)

        # State
        self._polys = []
        self._offset = np.array([0, 0])
        self._poly_btn.setChecked(True)
        self._chull_box.setChecked(True)
        self._scale_sld.setData(0.8, 1.2, 1.0)

        # Listeners
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn, self._chull_box]:
            btn.toggled.connect(self._recompute)
        self._touchpad.moved.connect(self._on_touchpad_move)
        self._scale_sld.valueChanged.connect(lambda _: self._recompute())

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
        scale = self._scale_sld.value()
        rois = []
        for poly in self._polys:
            poly = poly * scale + self._offset * self.MOVE_SCALE
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

    def _on_touchpad_move(self, dx: np.ndarray):
        self._offset += dx
        self._recompute()

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