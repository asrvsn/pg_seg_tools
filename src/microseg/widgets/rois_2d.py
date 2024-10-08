''' 2D widgets '''

from typing import Tuple, List, Callable
import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsItem
import pyqtgraph as pg

from matgeo.plane import PlanarPolygon
from matgeo.ellipsoid import Circle, Ellipsoid

from .base import *
from microseg.utils import pg_colors

class SelectableItem:
    def __init__(self, label: int=0):
        self.label = label
        self._proxy = ClickProxy()
        self.sigClicked = self._proxy.sigClicked
        self._pen = pg_colors.cc_pens[label % pg_colors.n_pens]
        self._hpen = pg_colors.cc_pens_hover[label % pg_colors.n_pens]
        self._selected = False
        self.setPen(self._pen)
        self.setAcceptHoverEvents(True)

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

class LabeledThing(abc.ABC):
    def __init__(self, l: int):
        self.l = l

    @abc.abstractmethod
    def copy(self) -> 'LabeledThing':
        pass

    @abc.abstractmethod
    def flipy(self) -> 'LabeledThing':
        ''' This is needed because of Pyqtgraph's insane orientation defaults '''
        pass

    @abc.abstractmethod
    def __add__(self, offset: np.ndarray):
        ''' This is needed because of Pyqtgraph's insane orientation defaults '''
        pass

class LabeledPolygon(PlanarPolygon, LabeledThing):
    def __init__(self, l: int, vertices: np.ndarray, **kwargs):
        PlanarPolygon.__init__(self, vertices, **kwargs)
        LabeledThing.__init__(self, l)

    def copy(self) -> 'LabeledPolygon':
        return LabeledPolygon(self.l, self.vertices.copy())
    
    def __eq__(self, other: 'LabeledPolygon'):
        return self.l == other.l and np.allclose(self.vertices, other.vertices)

    def flipy(self, yval: float) -> 'LabeledPolygon':
        poly = super().flipy(yval)
        return LabeledPolygon(self.l, poly.vertices)

    @staticmethod
    def from_pointcloud(l: int, vertices: np.ndarray) -> 'LabeledPolygon':
        vertices = PlanarPolygon.from_pointcloud(vertices).vertices
        return LabeledPolygon(l, vertices, check=False)
    
    @staticmethod
    def from_poly(l: int, poly: PlanarPolygon) -> 'LabeledPolygon':
        return LabeledPolygon(l, poly.vertices)

class SelectablePolygonItem(QGraphicsPolygonItem, SelectableItem):

    def __init__(self, poly: LabeledPolygon, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(pg.QtGui.QPolygonF([
            QtCore.QPointF(p[0], p[1]) for p in poly.vertices
        ]), *args, label=poly.l, **kwargs)
        self._poly = poly

class LabeledCircle(Circle, LabeledThing):
    def __init__(self, l: int, v: np.ndarray, r: float):
        Circle.__init__(self, v, r)
        LabeledThing.__init__(self, l)

    def copy(self) -> 'LabeledCircle':
        return LabeledCircle(self.l, self.v.copy(), self.r)
    
    def __eq__(self, other: 'LabeledCircle') -> bool:
        return self.l == other.l and np.allclose(self.v, other.v) and np.allclose(self.M, other.M)

    def flipy(self, yval: float) -> 'LabeledCircle':
        circ = super().flipy(yval)
        return LabeledCircle(self.l, circ.v, circ.r)
    
    @staticmethod
    def from_ellipse(l: int, ell: Ellipsoid) -> 'LabeledCircle':
        circ = Circle.from_ellipse(ell)
        return LabeledCircle(l, circ.v, circ.r)

class SelectableCircleItem(QGraphicsEllipseItem, SelectableItem):

    def __init__(self, circ: LabeledCircle, *args, **kwargs):
        # Both constructors get called automatically, so pass the named one explicitly
        super().__init__(*args, label=circ.l, **kwargs) 
        self._circ = circ
        self.setRadius(circ.r)

    def setRadius(self, r: float):
        self.setRect(self._circ.v[0]-r, self._circ.v[1]-r, 2*r, 2*r)
        self._circ.r = r