''' 
2D ROIs
'''

from typing import Tuple, List, Callable
import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem
import pyqtgraph as pg
from abc import ABC, abstractmethod

from matgeo.plane import PlanarPolygon
from matgeo.ellipsoid import Circle, Ellipsoid, Ellipse

from .base import *
from microseg.utils import pg_colors

'''
ROIs
'''

ROI = Union[PlanarPolygon, Ellipse, Circle]

class LabeledROI:
    def __init__(self, lbl: int, roi: ROI):
        self.lbl = lbl
        self.roi = roi

    def copy(self) -> 'LabeledROI':
        return LabeledROI(self.lbl, self.roi.copy())

    def flipy(self, yval: float) -> 'LabeledROI':
        ''' This is needed because of Pyqtgraph's insane orientation defaults '''
        return LabeledROI(self.lbl, self.roi.flipy(yval))

    def __add__(self, offset: np.ndarray):
        ''' This is needed because of Pyqtgraph's insane orientation defaults '''
        return LabeledROI(self.lbl, self.roi + offset)
    
    def __sub__(self, offset: np.ndarray):
        ''' This is needed because of Pyqtgraph's insane orientation defaults '''
        return LabeledROI(self.lbl, self.roi - offset)
    
    def to_item(self, img_shape: Tuple[int, int]) -> 'LabeledROIItem':
        '''
        Converts to item and appropriately transforms to pyqtgraph's orientation defaults
        '''
        self = self.toPyQTOrientation(img_shape)
        if type(self.roi) is PlanarPolygon:
            return LabeledPolygonItem(self)
        elif type(self.roi) is Ellipse:
            return LabeledEllipseItem(self)
        elif type(self.roi) is Circle:
            return LabeledCircleItem(self)
        else:
            raise NotImplementedError
        
    def toPyQTOrientation(self, img_shape: Tuple[int, int]) -> 'LabeledROI':
        '''
        Transform appropriately for rendering in pyqtgraph's weird orientation defaults
        '''
        xlen, ylen = img_shape
        offset = np.array([0, ylen-xlen])
        return (self + offset).flipy(ylen)
    
    def fromPyQTOrientation(self, img_shape: Tuple[int, int]) -> 'LabeledROI':
        '''
        Transform appropriately for rendering in pyqtgraph's weird orientation defaults
        '''
        xlen, ylen = img_shape
        offset = np.array([0, ylen-xlen])
        return self.flipy(ylen) - offset
        
'''
ROI Items (for adding to Qt widgets)
'''

class LabeledROIItem:
    def __init__(self, lroi: LabeledROI, selectable: bool=True, show_label: bool=True):
        self._lroi = lroi
        self._selected = False
        self._set_pens(show_label=show_label)
        if selectable:
            self._proxy = ClickProxy()
            self.sigClicked = self._proxy.sigClicked
            self.setAcceptHoverEvents(True)

    def _set_pens(self, show_label: bool=True):
        i = self._lroi.lbl % pg_colors.n_pens if show_label else 65
        self._pen = pg_colors.cc_pens[i]
        self._hpen = pg_colors.cc_pens_hover[i]
        self.setPen(self._pen)

    def hoverEnterEvent(self, event):
        self.setPen(self._hpen)

    def hoverLeaveEvent(self, event):
        if not self._selected:
            self.setPen(self._pen)

    def mousePressEvent(self, event):
        self.sigClicked.emit()
        self.select()

    def select(self):
        self._selected = True
        self.setPen(self._hpen)

    def unselect(self):
        self._selected = False
        self.setPen(self._pen)

    def to_roi(self, img_shape: Tuple[int, int]) -> LabeledROI:
        '''
        Converts to ROI and appropriately transforms back from pyqtgraph's orientation defaults
        '''
        return self._lroi.fromPyQTOrientation(img_shape)

class LabeledPolygonItem(QGraphicsPolygonItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(pg.QtGui.QPolygonF([
            QtCore.QPointF(p[0], p[1]) for p in lroi.vertices
        ]), *args, lroi=lroi, **kwargs)

class LabeledEllipseItem(QGraphicsEllipseItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(*args, lroi=lroi, **kwargs)
        P, rs = lroi.get_axes_stretches()
        hr, vr = rs
        x, y = lroi.v
        self.setRect(x-hr, y-vr, 2*hr, 2*vr)
        P = P.T
        theta = np.arctan2(P[1,0], P[0,0]) # Angle from orthogonal matrix P.T
        self.setTransformOriginPoint(QtCore.QPointF(x+hr, y+vr)) # Transform about ellipse center
        self.setRotation(theta*180/np.pi) # Rotate onto basis P

class LabeledCircleItem(QGraphicsEllipseItem, LabeledROIItem):
    def __init__(self, lroi: LabeledROI, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(*args, lroi=lroi, **kwargs) 
        self.setRadius(lroi.r)

    def setRadius(self, r: float):
        self.setRect(self._lroi.v[0]-r, self._lroi.v[1]-r, 2*r, 2*r)
        self._lroi.r = r