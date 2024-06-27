'''
2D segmentation widgets
'''
from qtpy.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox
import cellpose
import cellpose.models
import cellpose.io
import skimage
import skimage.restoration as restoration

from .pg import *
from microseg.data.seg_2d import *

class ZProjectWidget(SaveableWidget):
    ''' 
    Z Projection widget for extracting z-projection from ZXY data 
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Widgets
        self._plot = ImagePlotWidget(title='Z Projection')
        self._main_layout.addWidget(self._plot)
        self._mode_dropdown = QtWidgets.QComboBox()
        self._mode_dropdown.addItems(AVAIL_ZPROJ_MODES)
        self._settings_layout.addWidget(self._mode_dropdown)
        self._int_slider = IntegerSlider(mode='scroll')
        self._rng_slider = IntegerRangeSlider()
        self._settings_layout.addWidget(self._int_slider)
        self._settings_layout.addWidget(self._rng_slider)
        self._window_label = QLabel()
        self._settings_layout.addWidget(self._window_label)
        self.setEnabled(False) # Initially disabled
        self._int_slider.hide()
        self._rng_slider.hide()
        # self._settings_layout.addStretch()

        # Listeners
        self._mode_dropdown.currentTextChanged.connect(self.setMode)
        self._save_btn.clicked.connect(lambda: self.saved.emit())
        self._int_slider.valueChanged.connect(self._on_slider_change)
        self._rng_slider.valueChanged.connect(self._on_slider_change)

        # State
        self._stack = None
        self._zproj = None
        self._window = None
        self._mode = 0
        self._zmax = None

    def setData(self, stack: np.ndarray, mode: str, window: Tuple[int, int]):
        self.setEnabled(True)
        # Check bounds
        z, x, y = stack.shape
        # Set viewport range if this is new data
        lim = max(x, y)
        self._plot.setXRange(0, lim)
        self._plot.setYRange(0, lim)
        # self._plot.setLimits(xMin=0, xMax=lim, yMin=0, yMax=lim)
        # Update state
        self._stack = stack
        assert mode in AVAIL_ZPROJ_MODES, f'Invalid zproj mode {mode}'
        # Check bounds
        z, x, y = stack.shape
        self._zmax = z - 1
        assert 0 <= window[0] <= window[1] <= self._zmax, 'zproj window must be within z range'
        # Update state
        self._window = window
        self.setMode(mode)

    def getData(self) -> Tuple[np.ndarray, str, Tuple[int, int]]:
        assert not self._zproj is None, 'Must set data before getting data'
        assert self._zproj.shape == self._stack.shape[1:], 'zproj shape must match stack shape'
        mode = AVAIL_ZPROJ_MODES[self._mode]
        return self._zproj.copy(), mode, self._window

    def setMode(self, mode: str):
        assert not self._stack is None, 'Must set data before setting mode'
        self._mode = AVAIL_ZPROJ_MODES.index(mode)
        self._mode_dropdown.setCurrentIndex(self._mode)
        self._save_btn.setDisabled(False)
        # Reify mode
        i, j = self._window
        if mode == 'slice':
            z = int((i + j) / 2)
            self._window = (z, z)
            self._int_slider.setData(0, self._zmax, z)
            self._int_slider.show()
            self._rng_slider.hide()
        else:
            if i == j:
                if i == 0:
                    j = 1
                elif i == self._zmax:
                    i = self._zmax - 1
                else:
                    j = i + 1
                self._window = (i, j)
            self._rng_slider.setData(0, self._zmax, self._window)
            self._int_slider.hide()
            self._rng_slider.show()
        # Recompute zproj
        self._recompute()

    def _recompute(self):
        self._zproj = self._process(self._stack)
        self._plot._img.setImage(self._zproj)

    def _process(self, stack: np.ndarray) -> np.ndarray:
        # Perform Z projection
        mode = AVAIL_ZPROJ_MODES[self._mode]
        i, j = self._window
        if mode == 'slice':
            zproj = stack[i]
        else:
            j_ = j + 1
            if mode == 'max':
                zproj = stack[i:j_].max(axis=0)
            elif mode == 'min':
                zproj = stack[i:j_].min(axis=0)
            elif mode == 'mean':
                zproj = stack[i:j_].mean(axis=0)
            elif mode == 'median':
                zproj = np.median(stack[i:j_], axis=0)
            else:
                raise Exception(f'Invalid zproj mode {mode}')
        return zproj
    
    def _on_slider_change(self, value: Union[int, Tuple[int, int]]):
        if isinstance(value, int):
            self._window = (value, value)
        else:
            self._window = value
        self._recompute()

class ZStackWidget(ZProjectWidget):
    '''
    Slider-only Z project widget
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_dropdown.hide()
    
    def setData(self, stack: np.ndarray, index: int):
        super().setData(stack, 'slice', (index, index))

class CellposeWidget(SaveableWidget):
    '''
    Image/Mask widget item providing cellpose segmentation
    '''
    n_channels = 3
    use_gpu = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._plot = MaskImageWidget(editable=False)
        self._plot.setTitle('Cellpose Segmentation')
        self._main_layout.addWidget(self._plot)
        self._settings_layout.addWidget(QLabel('Dn:'))
        self._denoise_dropdown = QComboBox()
        self._denoise_dropdown.addItems(AVAIL_DENOISE_MODES)
        self._settings_layout.addWidget(self._denoise_dropdown)
        self._settings_layout.addWidget(QLabel('M:'))
        self._model_dropdown = QComboBox()
        self._model_dropdown.addItems(AVAIL_CP_MODELS)
        self._settings_layout.addWidget(self._model_dropdown)
        self._settings_layout.addWidget(QLabel('D:'))
        self._diam_box = QSpinBox()
        self._diam_box.setMinimum(1)
        self._diam_box.setMaximum(1000)
        self._diam_disk = QGraphicsEllipseItem(0,0,0,0)
        self._diam_disk.setPen(pg.mkPen('r'))
        self._diam_disk.setBrush(pg.mkBrush(255,0,0,77))
        self._plot.addItem(self._diam_disk)
        self._settings_layout.addWidget(self._diam_box)
        self._settings_layout.addWidget(QLabel('F:'))
        self._flow_box = QDoubleSpinBox()
        self._settings_layout.addWidget(self._flow_box)
        self._settings_layout.addWidget(QLabel('Cp:'))
        self._cellprob_box = QDoubleSpinBox()
        self._cellprob_box.setMinimum(-6)
        self._cellprob_box.setMaximum(6)
        self._settings_layout.addWidget(self._cellprob_box)
        self._settings_layout.addWidget(QLabel('NC:'))
        self._nuclear_dropdown = QComboBox()
        self._nuclear_dropdown.addItem('None')
        for c in range(self.n_channels):
            self._nuclear_dropdown.addItem(f'Chan {c}')
        self._settings_layout.addWidget(self._nuclear_dropdown)
        self._calib_btn = PushButton('Calibrate')
        self._settings_layout.addWidget(self._calib_btn)
        self._run_btn = PushButton('Run')
        self._settings_layout.addWidget(self._run_btn)
        self._settings_layout.addStretch()

        # Add tickbox to hide settings
        self._bottom_layout.addWidget(QLabel('Settings:'))
        self._settings_chk = QtWidgets.QCheckBox()
        self._settings_chk.setChecked(True)
        self._bottom_layout.addWidget(self._settings_chk)

        # Listeners
        self._denoise_dropdown.currentTextChanged.connect(self._on_denoise)
        self._model_dropdown.currentTextChanged.connect(self._on_model_change)
        self._diam_box.valueChanged.connect(self._on_diam_change)
        self._flow_box.valueChanged.connect(self._on_flow_change)
        self._cellprob_box.valueChanged.connect(self._on_cellprob_change)
        self._calib_btn.clicked.connect(self._on_calibrate)
        self._nuclear_dropdown.currentTextChanged.connect(self._on_nuclear_change)
        self._run_btn.clicked.connect(self._on_run)
        self._settings_chk.stateChanged.connect(lambda _: self._settings_widget.show() if self._settings_chk.isChecked() else self._settings_widget.hide())

        # State
        self._orig_img = None
        self._cxy_img = None
        self._channel = None
        self._model = AVAIL_CP_MODELS[0]
        self._nuclear_channel = None
        self._diam = 30
        self._denoise = 0
        self._mask = None
        self._flow = 0.4
        self._cellprob = 0.0
        self._cp_logger = cellpose.io.logger_setup()

        self._draw_settings()

    def _draw_settings(self):
        self._model_dropdown.setCurrentIndex(AVAIL_CP_MODELS.index(self._model))
        self._denoise_dropdown.blockSignals(True) # Prevent signal from firing
        self._denoise_dropdown.setCurrentIndex(self._denoise)
        self._denoise_dropdown.blockSignals(False)
        self._diam_box.setValue(self._diam)
        self._flow_box.setValue(self._flow)
        self._cellprob_box.setValue(self._cellprob)
        self._draw_diam()
        if self._nuclear_channel is None:
            self._nuclear_dropdown.setCurrentIndex(0)
        else:
            self._nuclear_dropdown.setCurrentIndex(self._nuclear_channel + 1)

    def _draw_diam(self):
        if not self._cxy_img is None:
            x, y = self._cxy_img.shape[1:]
            x, y = x//2, y//2
            r = self._diam//2
            self._diam_disk.setRect(x-r, y-r, self._diam, self._diam)

    def _draw(self):
        self._plot.setData(self._cxy_img[self._channel], self._mask)

    def _on_model_change(self, model: str):
        self._model = model

    def _on_diam_change(self, diam: int):
        self._diam = diam
        self._draw_diam()

    def _on_flow_change(self, flow: float):
        self._flow = flow

    def _on_cellprob_change(self, cellprob: float):
        self._cellprob = cellprob

    def _on_nuclear_change(self, _: str):
        idx = self._nuclear_dropdown.currentIndex()
        self._nuclear_channel = idx - 1 if idx > 0 else None

    def _validate_data(self):
        assert not self._cxy_img is None, 'Must set data before running'
        assert not self._channel is None, 'Must set channel before running'

    def _validate_cp(self):
        # Validate state
        self._validate_data()
        assert self._model in AVAIL_CP_MODELS, 'Invalid model'
        assert self._nuclear_channel is None or self._nuclear_channel < self.n_channels, 'Invalid nuclear channel'
        assert self._channel != self._nuclear_channel, 'Nuclear channel must be different from main channel'
        # Validate use of nuclear channel
        if self._model == 'nuclei':
            assert self._nuclear_channel is None, 'Cannot use nuclear channel with nuclei model'
        # Validate diameter
        assert self._diam > 0, 'Diameter must be positive'

    def _get_model(self) -> cellpose.models.Cellpose:
        return cellpose.models.Cellpose(model_type=self._model, gpu=self.use_gpu)

    def _on_calibrate(self):
        self._validate_cp()
        print('CP calibrate...')
        model = self._get_model()
        diam, _ = model.sz.eval(self._cxy_img[self._channel].copy())
        print('CP calibrate got diam', diam)
        self._diam = round(diam)
        self._draw_settings()

    def _on_denoise(self):
        self._denoise = self._denoise_dropdown.currentIndex()
        self._validate_data()
        denoise = AVAIL_DENOISE_MODES[self._denoise]
        img = self._orig_img[self._channel]
        print('CP denoise...', denoise)
        if denoise != 'None':
            if denoise == 'Bi':
                denoise_fun = restoration.denoise_bilateral
            elif denoise == 'TV_B':
                denoise_fun = restoration.denoise_tv_bregman
            elif denoise == 'Wvt':
                denoise_fun = restoration.denoise_wavelet
            else:
                raise Exception(f'Invalid denoise mode {denoise}')
            img = restoration.denoise_invariant(img, denoise_function=denoise_fun)
            img = skimage.img_as_uint(img)
        self._cxy_img[self._channel] = img
        print('CP denoise finished')
        self._draw()

    def _on_run(self):
        self._validate_cp()
        print('CP run...', self._model, self._channel, self._nuclear_channel, self._diam, self._flow, self._cellprob)
        model = self._get_model()
        chans = [self._channel+1, 0]
        if not self._nuclear_channel is None:
            chans[1] = self._nuclear_channel + 1
        img = self._cxy_img.transpose((1,2,0)) # Cellpose expects XYC
        self._mask = model.eval(img, diameter=self._diam, channels=chans, flow_threshold=self._flow, cellprob_threshold=self._cellprob)[0]
        print('CP run finished')
        self._draw()

    def setData(self, 
            orig_img: np.ndarray,
            cxy_img: np.ndarray,
            channel: int,
            mask: np.ndarray,
            denoise: str,
            model: str,
            nuclear_channel: Optional[int],
            diam: int,
            flow: float,
            cellprob: float,
        ):
        assert cxy_img.ndim == 3, 'Image must be CXY'
        # assert cxy_img.shape[0] == self.n_channels, 'Image must have same number of channels as widget'
        assert orig_img.shape == cxy_img.shape, 'Image must have same shape as widget'
        assert channel < self.n_channels, 'Channel must be within number of channels'
        assert mask.ndim == 2, 'Mask must be XY'
        assert mask.shape == cxy_img.shape[1:], 'Mask must have same shape as image'
        assert denoise in AVAIL_DENOISE_MODES, f'Invalid denoise mode {denoise}'
        assert model in AVAIL_CP_MODELS, f'Invalid model {model}'
        self.setEnabled(True)
        self._orig_img = orig_img
        self._cxy_img = cxy_img
        self._channel = channel
        self._denoise = AVAIL_DENOISE_MODES.index(denoise)
        self._model = model
        self._nuclear_channel = nuclear_channel
        self._diam = diam
        self._flow = flow
        self._cellprob = cellprob
        self._mask = mask
        self._draw_settings()
        self._draw()

    def getData(self) -> Tuple[np.ndarray, np.ndarray, str, str, Optional[int], int, float, float]:
        img = self._cxy_img[self._channel]
        mask = self._mask.copy()
        denoise = AVAIL_DENOISE_MODES[self._denoise]
        M = self._model
        NC = self._nuclear_channel
        D = self._diam
        F = self._flow
        Cp = self._cellprob
        return img, mask, denoise, M, NC, D, F, Cp

class CuratorWidget(SaveableWidget):
    '''
    Manual segmentation curator of automated segmentations
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._plot = MaskImageWidget(editable=True)
        self._plot.setTitle('Curated Segmentation')
        self._main_layout.addWidget(self._plot)
        self._plot_outline = MaskImageWidget(editable=True)
        self._plot_outline.setTitle('Curated Outline')
        self._main_layout.addWidget(self._plot_outline)
        # Hide outline by default
        self._plot_outline.hide()
        self._showing_dropdown = QtWidgets.QComboBox()
        self._showing_dropdown.addItems(['Mask', 'Outline'])
        self._settings_layout.addWidget(self._showing_dropdown)
        self._reset_orig_btn = PushButton('Reset to original')
        self._settings_layout.addWidget(self._reset_orig_btn)
        self._reset_saved_btn = PushButton('Reset to saved')
        self._settings_layout.addWidget(self._reset_saved_btn)
        self._settings_layout.addStretch()

        # Listeners
        self._showing_dropdown.currentTextChanged.connect(lambda _: self._on_change_view())
        self._reset_orig_btn.clicked.connect(lambda: self._on_reset_orig())
        self._reset_saved_btn.clicked.connect(lambda: self._on_reset_saved())

        # State
        self._img = None
        self._mask_orig = None
        self._mask_prev = None

    def setData(self, img: np.ndarray, mask_orig: np.ndarray, mask: np.ndarray, mask_outline: np.ndarray):
        assert img.shape == mask_orig.shape == mask.shape, 'Image and masks must have same shape'
        assert img.shape == mask_outline.shape, 'Image and outline mask must have same shape'
        self.setEnabled(True)
        self._img = img
        self._mask_orig = mask_orig
        self._mask_prev = mask.copy()
        self._plot.setData(img, mask)
        self._plot_outline.setData(img, mask_outline)
        # Set to mask view by default
        self._showing_dropdown.setCurrentIndex(0)

    def getData(self) -> Tuple[np.ndarray, np.ndarray]:
        mask = self._plot._mask_item._mask # TODO: fix this
        outline_mask = self._plot_outline._mask_item._mask # TODO: fix this
        return mask, outline_mask
    
    def _on_reset_orig(self):
        print('reset orig')
        self._plot.setData(self._img, self._mask_orig.copy())
    
    def _on_reset_saved(self):
        print('reset saved')
        self._plot.setData(self._img, self._mask_prev.copy())

    def _on_change_view(self):
        view = self._showing_dropdown.currentText()
        if view == 'Mask':
            self._plot.show()
            self._plot_outline.hide()
        elif view == 'Outline':
            self._plot.hide()
            self._plot_outline.show()
        else:
            raise Exception(f'Invalid view {view}')
        