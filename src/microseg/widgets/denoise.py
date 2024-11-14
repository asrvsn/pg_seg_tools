'''
Widget for denoising
'''
import numpy as np
import skimage
import skimage.restoration as skrest
from qtpy.QtWidgets import QRadioButton
from qtpy.QtCore import Signal

from .base import *

class DenoiseWidget(HLayoutWidget):
    OPTIONS = [
        ('none', None),
        ('tvb', skrest.denoise_tv_bregman),
        ('bi', skrest.denoise_bilateral),
        ('wvt', skrest.denoise_wavelet),
    ]
    processed = Signal(np.ndarray)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._img = None
        self._bts = []
        for opt in self.OPTIONS:
            btn = QRadioButton(opt)
            self.addWidget(btn)
            self._bts.append(btn)
            btn.toggled.connect(self._recalculate)
    
    def _recalculate(self):
        assert not self._img is None
        i = self._bts.index(self.sender())
        fn = self.OPTIONS[i][1]
        if fn is None:
            img = self._img.copy()
        else:
            img = skrest.denoise_invariant(self._img.copy(), fn)
        self.processed.emit(img)

    ''' Public methods '''

    def setImage(self, img: np.ndarray):
        self._img = img
        self._recalculate()