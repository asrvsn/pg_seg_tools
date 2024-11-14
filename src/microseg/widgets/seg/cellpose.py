'''
Cellpose3-based segmentor
'''
import numpy as np
from qtpy.QtWidgets import QCheckBox, QComboBox, QLabel, QPushButton, QRadioButton, QButtonGroup, QLineEdit, QSpinBox
from qtpy.QtGui import QIntValidator
import cellpose
import cellpose.models
import upolygon
from scipy.ndimage import find_objects
import cv2
import pdb
import skimage
import skimage.restoration as skrest

from matgeo import PlanarPolygon, Circle, Ellipse
from microseg.widgets.pg import ImagePlotWidget
from microseg.utils.image import rescale_intensity
import microseg.utils.mask as mutil
from .base import *
from .manual import ROICreatorWidget

class ImageProcessingWidget(VLayoutWidget):
    FILTERS = [
        ('none', None),
        ('tvb', skrest.denoise_tv_bregman),
        ('bi', skrest.denoise_bilateral),
        ('wvt', skrest.denoise_wavelet),
    ]
    processed = QtCore.Signal()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._intens_box = QCheckBox('Rescale intensity')
        self.addWidget(self._intens_box)
        down_wdg = HLayoutWidget()
        self._down_box = QCheckBox('Downscale to (px):')
        down_wdg.addWidget(self._down_box)
        down_wdg.addSpacing(10)
        self._down_int = QSpinBox(minimum=100, maximum=10000)
        down_wdg.addWidget(self._down_int)
        self.addWidget(down_wdg)
        self.addWidget(QLabel('Denoise:'))
        self._filt_btns = []
        for (name,_) in self.FILTERS:
            btn = QRadioButton(name)
            self.addWidget(btn)
            self._filt_btns.append(btn)

        # State
        self._img = None
        self._processed_img = None
        self._down_box.setChecked(False)
        self._down_int.setValue(400)
        self._intens_box.setChecked(False)
        self._filt_btns[0].setChecked(True)

        # Listeners
        for btn in self._filt_btns + [
                self._down_box, self._intens_box,    
            ]:
            btn.toggled.connect(self._recalculate)
        self._down_int.valueChanged.connect(self._recalculate)
    
    def _recalculate(self):
        assert not self._img is None
        img = self._img.copy()
        ## Intensity
        if self._intens_box.isChecked():
            img = rescale_intensity(img)
        ## Downsampling
        if self._down_box.isChecked():
            down_n = self._down_int.value()
            if down_n < max(img.shape):
                img = skimage.transform.rescale(img, self.scale, anti_aliasing=True)
        ## Denoising
        for i, btn in enumerate(self._filt_btns):
            if btn.isChecked():
                dn_fn = self.FILTERS[i][1]
                if not dn_fn is None:
                    dnkw = dict(channel_axis=-1) if img.ndim == 3 else {}
                    img = skrest.denoise_invariant(img, dn_fn, denoiser_kwargs=dnkw)
                break
        ## Emit
        self._processed_img = img
        self.processed.emit()

    ''' Public '''

    @property
    def scale(self) -> float:
        return self._down_int.value() / max(self._img.shape) if self._down_box.isChecked() else 1
    
    @property
    def processed_img(self) -> np.ndarray:
        return self._processed_img
    
    def setImage(self, img: np.ndarray):
        self._img = img
        self._recalculate()

class CellposeMultiSegmentorWidget(SegmentorWidget):
    USE_GPU: bool=False
    MODELS: List[str] = [
        'cyto3',
        'nuclei',
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        ## For image settings
        self._img_wdg = VGroupBox('Image settings')
        self._main.addWidget(self._img_wdg)
        self._img_proc = ImageProcessingWidget()
        self._img_wdg.addWidget(self._img_proc)
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

        # State
        self._set_cp_model(0)
        self._cp_cellprob_sld.setData(-3, 4, 0.)

        # Listeners
        self._cp_mod_drop.currentIndexChanged.connect(self._set_cp_model)
        self._cp_btn.clicked.connect(self._recompute)

    ''' Overrides '''

    def name(self) -> str:
        return 'Cellpose (multi)'

    def make_proposals(self, img: np.ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        ''' 
        Recomputes only the mask/poly post-processing step if no existing cellpose mask exists.
        Cellpose mask is re-computed only on explicit user request.
        '''
        if self._img_proc.processed_img is None:
            self._update_img(img, poly)
        if self._cp_polys is None:
            self._update_cp_polys()
        return self._cp_polys

    def reset_state(self):
        super().reset_state()
        self._cp_polys = None
        if hasattr(self, '_cp_cellprob_sld'):
            self._cp_cellprob_sld.setValue(0.)

    def keyPressEvent(self, evt):
        if evt.key() == QtCore.Qt.Key_Return and evt.modifiers() & Qt.ShiftModifier:
            self._recompute()
        else:
            super().keyPressEvent(evt)

    ''' Private methods '''

    def _set_cp_model(self, idx: int):
        '''
        Sets the cellpose model
        '''
        self._cp_model = cellpose.models.Cellpose(
            model_type=self.MODELS[idx],
            gpu=self.USE_GPU
        )

    def _compute_cp_polys(self, img: np.ndarray, scale: float, poly: PlanarPolygon) -> List[PlanarPolygon]:
        '''
        Compute cellpose polygons using with possible downscaling
        Returns polygons in the original (un-downscaled) coordinate system
        '''
        assert scale > 0
        # diam = poly.circular_radius() * 2
        poly = poly.set_res(scale, scale)
        diam = poly.diameter()
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
            _, contours, __ = upolygon.find_contours(i_mask.astype(np.uint8))
            contours = [np.array(c).reshape(-1, 2) for c in contours] # Convert X, Y, X, Y,... to X, Y
            contour = max(contours, key=lambda c: cv2.contourArea(c)) # Find max-area contour
            if contour.shape[0] < 3:
                continue
            contour = contour + np.array([sc.start, sr.start])
            try:
                poly = PlanarPolygon(contour)
                poly = poly.set_res(1/scale, 1/scale)
                cp_polys.append(poly)
            except:
                pass
        print(f'Found {len(cp_polys)} valid polygons')
        return cp_polys

    def _update_img(self, img: np.ndarray, poly: PlanarPolygon):
        self._img_proc.setImage(img)

    def _update_cp_polys(self):
        '''
        Computes & sets cellpose mask
        '''
        assert not self._poly is None and not self._img_proc.processed_img is None
        self._cp_polys = self._compute_cp_polys(self._img_proc.processed_img, self._img_proc.scale, self._poly)
    
    def _recompute(self):
        '''
        Recomputes entire thing
        '''
        self._update_cp_polys()
        self._set_proposals(self._cp_polys)


class CellposeSingleSegmentorWidget(CellposeMultiSegmentorWidget):
    '''
    Segment a single object by zooming in 
    '''
    WIN_MULT: float=1.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subimg_view = QImageWidget()
        self._main.insertWidget(0, self._subimg_view)
        self._drawing_box = QCheckBox('Show drawing')
        self._main.insertWidget(1, self._drawing_box)

        # State
        self._drawing_box.setChecked(False)

        # Listeners
        self._drawing_box.toggled.connect(lambda: self._render_img(self._poly))
        self._img_proc.processed.connect(lambda: self._render_img(self._poly))

    def name(self) -> str:
        return 'Cellpose (single)'
    
    def _update_img(self, img: np.ndarray, poly: PlanarPolygon):
        center = poly.centroid() 
        radius = np.linalg.norm(poly.vertices - center, axis=1).max() * self.WIN_MULT
        # Select image by center +- radius 
        xmin = max(0, math.floor(center[0] - radius))
        xmax = min(img.shape[1], math.ceil(center[0] + radius))
        ymin = max(0, math.floor(center[1] - radius))
        ymax = min(img.shape[0], math.ceil(center[1] + radius))
        # Store offset
        self._offset = np.array([xmin, ymin])
        self._center = np.array([xmax - xmin, ymax - ymin]) / 2
        subimg = img[ymin:ymax, xmin:xmax].copy()
        super()._update_img(subimg, poly)
        self._render_img(poly)
    
    def _render_img(self, poly: PlanarPolygon):
        subimg = self._img_proc.processed_img.copy()
        scale = self._img_proc.scale
        offset = self._offset
        # Render
        ar = subimg.shape[0] / subimg.shape[1]
        self._subimg_view.setFixedSize(220, round(220 * ar))
        if self._drawing_box.isChecked():
            subimg = mutil.draw_outline(subimg, (poly - offset).set_res(scale, scale))
        self._subimg_view.setImage(subimg)  
    
    def _compute_cp_polys(self, subimg: np.ndarray, scale: float, poly: PlanarPolygon) -> List[PlanarPolygon]:
        # Compute cellpose on sub-img & translate back
        polys = super()._compute_cp_polys(subimg, scale, poly - self._offset)
        if len(polys) > 0:
            poly = min(polys, key=lambda p: np.linalg.norm(p.centroid() - self._center))
            return [poly + self._offset]
        else:
            return []