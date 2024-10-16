'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton
import cellpose
import cellpose.models

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
        self._cp_wdg.addWidget(mod_wdg)
        self._cp_btn = QPushButton('Recompute')
        self._cp_wdg.addWidget(self._cp_btn)

        ## For mask / polygon postprocessing
        self._poly_wdg = VGroupBox('Polygon settings')
        self._main.addWidget(self._poly_wdg)
        self._chull_box = QCheckBox('Convex hull')
        self._poly_wdg.addWidget(self._chull_box)

        # State
        self._cp_mask = None
        self._set_cp_model(0)
        self._chull_box.setChecked(False)

        # Listeners
        for btn in [self._chull_box]:
            btn.toggled.connect(self._propose)
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)
        self._cp_btn.clicked.connect(self._recompute)

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose3'

    def make_proposals(self, poly: PlanarPolygon) -> List[ROI]:
        ''' 
        Recomputes only the mask/poly post-processing step if no existing cellpose mask exists.
        Cellpose mask is re-computed only on explicit user request.
        '''
        if self._cp_mask is None:
            self._cp_mask = self._compute_cp_mask(poly)
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

    def _compute_cp_mask(self, poly: PlanarPolygon) -> np.ndarray:
        '''
        Computes cellpose mask
        '''
        pass

    def _postprocess(self, cp_mask: np.ndarray) -> List[ROI]:
        '''
        Post-processes cellpose mask
        '''
        pass

    def _recompute(self):
        '''
        Recomputes entire thing
        '''
        assert not self._poly is None
        self._cp_mask = self._compute_cp_mask(self._poly)
        self._propose()