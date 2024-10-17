'''
Manual segmentor dialog
'''
from numpy import ndarray
from microseg.widgets.base import List
from .base import *

class ManualSegmentorWidget(SegmentorWidget):
    def name(self) -> str:
        return 'Manual'
    
    def make_proposals(self, img: ndarray, poly: PlanarPolygon) -> List[PlanarPolygon]:
        return [poly]