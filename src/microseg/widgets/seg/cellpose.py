'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup
import cellpose
import cellpose.models
import upolygon
from scipy.ndimage import find_objects
import cv2
import pdb

from matgeo import PlanarPolygon, Circle, Ellipse
from .base import *
from .manual import ROICreatorWidget

class CellposeMultiSegmentorWidget(SegmentorWidget):
    USE_GPU: bool=False
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
        cellprob_wdg = HLayoutWidget()
        cellprob_wdg.addWidget(QLabel('Cellprob:'))
        self._cp_cellprob_sld = FloatSlider(step=0.1)
        cellprob_wdg.addWidget(self._cp_cellprob_sld)
        self._cp_wdg.addWidget(cellprob_wdg)
        self._cp_btn = QPushButton('Recompute')
        self._cp_wdg.addWidget(self._cp_btn)
        self._main.addSpacing(10)

        ## For polygon -> ROI postprocessing
        self._roi_wdg = VGroupBox('ROI settings')
        self._main.addWidget(self._roi_wdg)
        self._roi_creator = ROICreatorWidget()
        self._roi_wdg.addWidget(self._roi_creator)

        # State
        self._set_cp_model(0)
        self._cp_cellprob_sld.setData(-3, 4, 0.)

        # Listeners
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)
        self._cp_btn.clicked.connect(self._recompute)
        self._roi_creator.edited.connect(self.propose.emit) # Bubble from the editor

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose (multi)'

    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[ROI]:
        ''' 
        Recomputes only the mask/poly post-processing step if no existing cellpose mask exists.
        Cellpose mask is re-computed only on explicit user request.
        '''
        if self._cp_polys is None:
            self._update_cp_polys(img, poly)
        self._roi_creator.setData(self._cp_polys)

    def reset_state(self):
        super().reset_state()
        self._cp_polys = None
        if hasattr(self, '_cp_cellprob_sld'):
            self._cp_cellprob_sld.setValue(0.)

    ''' Private methods '''

    def _set_cp_model(self, idx: int):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.Cellpose(
            model_type=self.MODELS[idx],
            gpu=self.USE_GPU
        )

    def _compute_cp_polys(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        diam = poly.circular_radius() * 2
        cellprob = self._cp_cellprob_sld.value()
        mask = self._cp_model.eval(
            img,
            diameter=diam,
            cellprob_threshold=cellprob,
        )[0]
        assert mask.shape == img.shape[:2]
        cp_polys = []
        slices = find_objects(mask)
        for i, si in enumerate(slices):
            if si is None:
                continue
            sr, sc = si
            i_mask = mask[sr, sc] == (i+1)
            # pdb.set_trace()
            _, contours, __ = upolygon.find_contours(i_mask.astype(np.uint8))
            contours = [np.array(c).reshape(-1, 2) for c in contours] # Convert X, Y, X, Y,... to X, Y
            contour = max(contours, key=lambda c: cv2.contourArea(c)) # Find max-area contour
            if contour.shape[0] < 3:
                continue
            contour = contour + np.array([sc.start, sr.start])
            poly = PlanarPolygon(contour)
            cp_polys.append(poly)
        return cp_polys

    def _update_cp_polys(self, img: np.ndarray, poly: PlanarPolygon):
        '''
        Computes & sets cellpose mask
        '''
        self._cp_polys = self._compute_cp_polys(img, poly)
    
    def _recompute(self):
        '''
        Recomputes entire thing
        '''
        assert not self._poly is None and not self._img is None
        self._update_cp_polys(self._img, self._poly)
        self._propose()


class CellposeSingleSegmentorWidget(CellposeMultiSegmentorWidget):
    '''
    Segment a single object by zooming in 
    '''
    WIN_MULT: float=1.5

    def name(self) -> str:
        return 'Cellpose (single)'
    
    def _compute_cp_polys(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        center = poly.centroid()
        radius = np.linalg.norm(poly.vertices - center, axis=1).max() * self.WIN_MULT
        # Select image by center +- radius 
        xmin = max(0, math.floor(center[0] - radius))
        xmax = min(img.shape[1], math.ceil(center[0] + radius))
        ymin = max(0, math.floor(center[1] - radius))
        ymax = min(img.shape[0], math.ceil(center[1] + radius))
        img = img[ymin:ymax, xmin:xmax]
        # Compute cellpose on sub-img & translate back
        offset = np.array([xmin, ymin])
        polys = super()._compute_cp_polys(img, poly - offset)
        if len(polys) > 0:
            poly = min(polys, key=lambda p: np.linalg.norm(p.centroid() + offset - center))
            return [poly + offset]
        else:
            return []