''' 2D widgets '''

from typing import Tuple, List, Callable
import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QGraphicsPolygonItem, QGraphicsRectItem, QGraphicsEllipseItem
import pyqtgraph as pg

from matgeo.plane import PlanarPolygon
from matgeo.ellipsoid import Sphere

from .base import *
from microseg.utils import pg_colors

class PolygonItem(QGraphicsPolygonItem):
    '''
    Polygon from numpy array
    '''
    def __init__(self, polygon: PlanarPolygon, color: np.ndarray, alpha: float=1.0, **kwargs):
        self.polygon = polygon
        self._brush = pg.mkBrush(*color, int(alpha*255))
        self._dark_brush = pg.mkBrush(*color, int(min(1,alpha*1.5)*255))
        super().__init__(pg.QtGui.QPolygonF([
            QtCore.QPointF(p[1], p[0]) for p in polygon.vertices
        ]), **kwargs)
        self.setBrush(self._brush)
        self.setAcceptHoverEvents(True)

    # Increase alpha on hover event
    def hoverEnterEvent(self, event):
        self.setBrush(self._dark_brush)

    # Decrease alpha on hover event
    def hoverLeaveEvent(self, event):
        self.setBrush(self._brush)

    # Print on click event
    def mousePressEvent(self, event):
        self.setBrush(self._dark_brush)
        super().mousePressEvent(event)

class TessellationItem(QGraphicsRectItem):
    '''
    Transparent layer containing multiple polygons as widget
    '''
    def __init__(self, polygons: List[PlanarPolygon], cmap: Callable=None, alpha: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self.polygons = polygons
        if cmap is None:
            cmap = lambda polys: map_colors(np.arange(len(polys)), 'categorical', i255=True)
        self.cmap = cmap
        self.alpha = alpha
        self.setAcceptHoverEvents(True)
        self.draw_polygons()

    def draw_polygons(self):
        colors = self.cmap(self.polygons)
        self.poly_widgets = []
        for p, c in zip(self.polygons, colors):
            pitem = PolygonItem(p, c, self.alpha)
            self.poly_widgets.append(pitem)
            pitem.setParentItem(self)

class LabeledCircle(Sphere):
    def __init__(self, l: int, v: np.ndarray, r: float):
        super().__init__(v, r)
        self.l = l

    def copy(self) -> 'LabeledCircle':
        return LabeledCircle(self.l, self.v.copy(), self.r)
    
    def __eq__(self, other: 'LabeledCircle') -> bool:
        return self.l == other.l and np.allclose(self.v, other.v) and np.allclose(self.M, other.M)

class LabeledCircleItem(QGraphicsEllipseItem):

    def __init__(self, circ: LabeledCircle):
        self._circ = circ
        super().__init__()
        self._proxy = ClickProxy()
        self.sigClicked = self._proxy.sigClicked
        self._pen = pg_colors.cc_pens[circ.l % pg_colors.n_pens]
        self._hpen = pg_colors.cc_pens_hover[circ.l % pg_colors.n_pens]
        self._selected = False
        self.setPen(self._pen)
        self.setAcceptHoverEvents(True)
        self.setRadius(circ.r)

    def setRadius(self, r: float):
        self.setRect(self._circ.v[0]-r, self._circ.v[1]-r, 2*r, 2*r)
        self._circ.r = r

    def hoverEnterEvent(self, event):
        self.setPen(self._hpen)

    def hoverLeaveEvent(self, event):
        if not self._selected:
            self.setPen(self._pen)

    def mousePressEvent(self, event):
        self.sigClicked.emit()
        self.select()

    def unselect(self):
        self._selected = False
        self.setPen(self._pen)

    def select(self):
        self._selected = True
        self.setPen(self._hpen)