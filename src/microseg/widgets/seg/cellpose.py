'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton
import cellpose
import cellpose.models
import upolygon

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *

class CellposeSegmentorWidget(SegmentorWidget):
    USE_GPU: bool=True
    MODELS: List[str] = [
        'cyto3',
        'nuclei',
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## For Cellpose settings
        self._cp_wdg = VGroupBox('Cellpose settings')
        self._main.addWidget(self._cp_wdg)
        mod_wdg = HLayoutWidget()
        mod_wdg.addWidget(QLabel('Model:'))
        self._cp_mod_drop = QComboBox()
        self._cp_mod_drop.addItems(self.MODELS)
        mod_wdg.addWidget(self._cp_mod_drop)
        cellprob_wdg = HLayoutWidget()
        cellprob_wdg.addWidget(QLabel('Cellprob:'))
        self._cp_cellprob_sld = FloatSlider(step=0.1)
        self._cp_wdg.addWidget(mod_wdg)
        self._cp_btn = QPushButton('Recompute')
        self._cp_wdg.addWidget(self._cp_btn)

        ## For mask -> ROI postprocessing
        self._roi_wdg = VGroupBox('ROI settings')
        self._main.addWidget(self._roi_wdg)
        self._poly_wdg = HLayoutWidget()
        self._poly_btn = QRadioButton('Make polygons')
        self._poly_wdg.addWidget(self._poly_btn)
        self._chull_box = QCheckBox('Convex hull')
        self._poly_wdg.addWidget(self._chull_box)
        self._roi_wdg.addWidget(self._poly_wdg)
        self._ellipse_btn = QRadioButton('Make ellipses')
        self._roi_wdg.addWidget(self._ellipse_btn)
        self._circle_btn = QRadioButton('Make circles')
        self._roi_wdg.addWidget(self._circle_btn)

        # State
        self._set_cp_model(0)
        self._cp_cellprob_sld.setData(-3, 4, 0.)
        self._chull_box.setChecked(False)

        # Listeners
        for btn in [self._poly_btn, self._ellipse_btn, self._circle_btn, self._chull_box]:
            btn.toggled.connect(self._roi_settings_update)
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)
        self._cp_btn.clicked.connect(self._recompute)

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose3'

    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[ROI]:
        ''' 
        Recomputes only the mask/poly post-processing step if no existing cellpose mask exists.
        Cellpose mask is re-computed only on explicit user request.
        '''
        if self._cp_mask is None:
            self._update_cp_mask(img, poly)
        return self._postprocess(self._cp_mask)

    def reset_state(self):
        super().reset_state()
        self._cp_mask = None

    ''' Private methods '''

    def _set_cp_model(self, idx: int):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.Cellpose(
            model_type=self.MODELS[idx],
            gpu=self.USE_GPU
        )

    def _update_cp_mask(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Computes & sets cellpose mask
        '''
        diam = poly.circular_radius() * 2
        cellprob = self._cp_cellprob_sld.value()
        mask = self._cp_model.eval(
            img,
            diameter=diam,
            cellprob_threshold=cellprob,
        )[0]
        assert mask.shape == img.shape[:2]
        self._cp_mask = mask
    
    def _postprocess(self, cp_mask: np.ndarray) -> List[ROI]:
        '''
        Post-processes cellpose mask into ROIs
        '''
        rois = []
        do_poly = self._poly_btn.isChecked()
        use_chull = self._chull_box.isChecked()
        do_ell = self._ellipse_btn.isChecked()
        do_circ = self._circle_btn.isChecked()
        labels = np.unique(cp_mask)
        for l in labels:
            if l == 0:
                continue
            l_mask = cp_mask == l
            _, contours, __ = upolygon.find_contours(l_mask.astype(np.uint8))
            contours = [np.array(c).reshape(-1, 2) for c in contours] # Convert X, Y, X, Y,... to X, Y
            contour = max(contours, key=lambda c: c.shape[0]) # Find longest contour
            poly = PlanarPolygon(contour)
            if do_poly:
                if use_chull: 
                    roi = poly.hullify()
                else:
                    roi = poly
            elif do_ell:
                roi = Ellipse.from_poly(poly)
            elif do_circ:
                roi = Circle.from_poly(poly)
            rois.append(roi)
        return rois

    def _recompute(self):
        '''
        Recomputes entire thing
        '''
        assert not self._poly is None and not self._img is None
        self._update_cp_mask(self._img, self._poly)
        self._propose()

    def _roi_settings_update(self):
        if self._poly_btn.isChecked():
            self._chull_box.setEnabled(True)
        else:
            self._chull_box.setEnabled(False)
        self._propose()