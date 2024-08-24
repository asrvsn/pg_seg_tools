'''
Manual polygon-based segmentor
'''

import pickle
import skimage
import skimage.io
import os

from matgeo.plane import PlanarPolygon
from .widgets.seg_2d import *
from .utils.data import *

class PolySegmentorWidget(SaveableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets   
        self._pliot = PolyImageWidget(editable=True)
        self._pliot.setTitle('Polygon Segmentation')
        self._main_layout.addWidget(self._pliot)

    def setData(self, img: np.ndarray, polys: List[PlanarPolygon]):
        self._plot.setData(img, polys)
        self.setEnabled(True)

    def getData(self) -> List[PlanarPolygon]:
        polys = [p.copy()]