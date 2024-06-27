'''
Pyqtgraph widgets
'''
from typing import Tuple, Optional, List, Callable
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut

import microseg.utils.mask as mutil
from microseg.utils.colors import *
from .rois_2d import *

'''
Functions
'''

def link_plots(p1: pg.PlotWidget, p2: pg.PlotWidget):
    p1.setXLink(p2)
    p1.setYLink(p2)
    p2.setXLink(p1)
    p2.setYLink(p1)

'''
Classes
'''

class NoTouchPlotWidget(pg.PlotWidget):
    '''
    Plot widget with touch events disabled
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't accept touch events due to bug on MacOS
        # https://bugreports.qt.io/browse/QTBUG-103935
        self.viewport().setAttribute(QtCore.Qt.WA_AcceptTouchEvents, False)

class PlotWidget(NoTouchPlotWidget):
    '''
    Plot widget with some extra listeners
    '''
    sigKeyPress = QtCore.Signal(object)

    def keyPressEvent(self, ev):
        # print('key press', ev.key())
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)

class ImagePlotWidget(NoTouchPlotWidget):
    '''
    Plot widget with default image item, zoom limits set to image
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAspectLocked(True)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self._img = pg.ImageItem()
        self.addItem(self._img)
        self.getViewBox().sigRangeChanged.connect(self._limit_zoom)

    def _limit_zoom(self):
        """Limit zoom to not zoom out beyond the image boundaries."""
        vb = self.getViewBox()
        x_range, y_range = vb.viewRange()
        img_shape = self._img.image.shape

        # Calculate the necessary range to keep the view within the image bounds
        new_x_range = [max(0, min(x_range[0], img_shape[1]-1)), min(img_shape[1], max(x_range[1], 0))]
        new_y_range = [max(0, min(y_range[0], img_shape[0]-1)), min(img_shape[0], max(y_range[1], 0))]

        # Set the new range
        vb.setRange(xRange=new_x_range, yRange=new_y_range, padding=0)

class MaskItem(pg.ImageItem):
    '''
    Image mask as widget
    '''
    def __init__(self, mask: np.ndarray=None, alpha: float=0.5, **kwargs):
        super().__init__(**kwargs)
        self._im = None
        self._cmap = lambda mask: map_colors(mask, 'mask', i255=True, alpha=alpha)
        self._alpha = alpha
        self._dark_alpha = min(1, alpha*1.7)
        self._hover_label = None
        self._click_label = None
        self._move_sensitive = True
        if not mask is None:
            self.setMask(mask)
    
    def setMask(self, mask: np.ndarray):
        self._mask = mask
        self._labels = np.unique(mask)
        self.draw_mask(recompute=True)

    def draw_mask(self, recompute: bool=False):
        if recompute:
            self._im = self._cmap(self._mask)
        im = self._im.copy()
        for dark_lbl in [self._click_label, self._hover_label]:
            if not dark_lbl is None and dark_lbl != 0:
                im[self._mask == dark_lbl,3] = int(self._dark_alpha * 255)
        self.setImage(im)

    def get_view_pos(self, pos) -> Tuple[float]:
        pos = self._plot._vb.mapSceneToView(pos)
        return round(pos.x()), round(pos.y())

    def get_label_at_pos(self, pos) -> Optional[int]:
        x, y = self.get_view_pos(pos)
        if 0 <= x < self._mask.shape[0] and 0 <= y < self._mask.shape[1]:
            return self._mask[x, y]
        else:
            return None

    def mouse_move(self, pos):
        if not QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
            old_label = self._hover_label
            self._hover_label = self.get_label_at_pos(pos)
            # print('move', self._hover_label)
            if old_label != self._hover_label:
                self.draw_mask()

    def mouseClickEvent(self, event):
        self._click_label = self.get_label_at_pos(event.scenePos())
        # print('click', self._click_label)
        self.draw_mask()

    def add_to_plot(self, plot: pg.PlotItem):
        self._plot = plot
        self._plot._vb = plot.getViewBox()
        plot.addItem(self)
        plot.scene().sigMouseMoved.connect(self.mouse_move)

class EditableMaskItem(MaskItem):
    '''
    Mask with editable labels
    '''
    undo_n: int=100
    
    def __init__(self, max_items: int=np.inf, *args, **kwargs):
        self._max_items = max_items
        self._shortcuts = []
        self._plot = None
        self._reset_drawing_state()
        super().__init__(*args, **kwargs)
    
    def setMask(self, mask: np.ndarray):
        self._undo_stack = [mask.copy()]
        self._redo_stack = []
        self._reset_drawing_state()
        super().setMask(mask)
    
    def _peek_mask(self):
        self._mask = self._undo_stack[-1].copy()
        self.draw_mask(recompute=True)
    
    def _update_mask(self, fun: Callable):
        mask = self._undo_stack[-1].copy()
        mask = fun(mask)
        self._undo_stack.append(mask)
        self._undo_stack = self._undo_stack[-self.undo_n:]
        self._redo_stack = []
        self._peek_mask()

    def _delete(self):
        print('delete')
        if not self._click_label is None:
            self._update_mask(lambda mask: mutil.delete_label(mask, self._click_label))
            self._click_label = None

    def _undo(self):
        print('undo')
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            self._peek_mask()
        else:
            print('Cannot undo further')

    def _redo(self):
        print('redo')
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            self._peek_mask()
        else:
            print('Cannot redo further')
    
    def _edit(self):
        print('edit')
        if not self._is_drawing:
            if np.unique(self._mask).shape[0] > self._max_items:
                print('Cannot draw more than', self._max_items, 'labels')
                return
            self._is_drawing = True
            self._drawn_label = self._mask.max() + 1
            # print('New drawn label:', self._drawn_label)
            # Disable viewport movement
            self._plot._vb.setMouseEnabled(x=False, y=False)

    def _stop_editing(self):
        if self._is_drawing:
            self._reset_drawing_state()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._drawn_poly = None
        self._drawn_label = None
        # Enable viewport movement
        if not self._plot is None:
            self._plot._vb.setMouseEnabled(x=True, y=True)

    def _enter(self):
        print('enter')
        self._stop_editing()

    def _escape(self):
        print('escape')
        self._stop_editing()
        self._click_label = None
        self._peek_mask()
    
    def _add_shortcut(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self._plot)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)
    
    def add_to_plot(self, plot: pg.PlotItem):
        super().add_to_plot(plot)
        # Add shortcuts
        self._add_shortcut('Delete', lambda: self._delete())
        self._add_shortcut('Backspace', lambda: self._delete())
        self._add_shortcut('Ctrl+Z', lambda: self._undo())
        self._add_shortcut('Ctrl+Y', lambda: self._redo())
        self._add_shortcut('Ctrl+Shift+Z', lambda: self._redo())
        self._add_shortcut('E', lambda: self._edit())
        self._add_shortcut('Enter', lambda: self._enter())
        self._add_shortcut('Escape', lambda: self._escape())

    def mouseClickEvent(self, event):
        if not self._is_drawing:
            super().mouseClickEvent(event)

    def mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                x, y = self.get_view_pos(pos)
                if self._drawn_poly is None:
                    print('starting draw')
                    self._drawn_poly = [y, x] # Reverse order
                else:
                    self._drawn_poly.append(y)
                    self._drawn_poly.append(x)
                    # Use self._mask as canvas
                    self._mask = mutil.draw_poly(self._mask, self._drawn_poly, self._drawn_label)
                    self.draw_mask(recompute=True)
            else:
                if not self._drawn_poly is None:
                    print('ending draw')
                    self._update_mask(lambda _: self._mask.copy())
                    self._stop_editing()
        else:
            super().mouse_move(pos)

class MaskImageWidget(NoTouchPlotWidget):
    '''
    Combined image + mask items
    '''
    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self.setAspectLocked(True)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self._editable = editable
        self._img_item = pg.ImageItem()
        self.addItem(self._img_item)
        self._mask_item = EditableMaskItem() if editable else MaskItem()
        self._mask_item.add_to_plot(self)

        # State
        self._img = None
        self._mask = None

    def setData(self, img: np.ndarray, mask: np.ndarray):
        assert img.shape == mask.shape, 'Image and mask must have same shape'
        self._img = img
        self._mask = mask
        self._img_item.setImage(img)
        self._mask_item.setMask(mask)

class CirclesImageWidget(ImagePlotWidget):
    '''
    Editable widget for drawing circles on an image
    '''
    edited = QtCore.Signal()
    
    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable
        
        # State
        self._circles: List[LabeledCircle] = []
        self._rois: List[pg.CircleROI] = []
        self._selected: int = None
        self._shortcuts = []
        self._reset_drawing_state()
        self._next_label: int = None

        # Listeners
        def add_sc(key: str, fun: Callable):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(fun)
            self._shortcuts.append(sc)
        add_sc('Delete', lambda: self._delete())
        add_sc('Backspace', lambda: self._delete())
        add_sc('E', lambda: self._edit())
        add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

    def setCircles(self, circles: List[LabeledCircle]):
        # Remove previous circles
        self._selected = None
        self._circles = circles
        for r in self._rois:
            self.removeItem(r)
        # Add new circles
        self._rois = []
        for i, circ in enumerate(circles):
            roi = LabeledCircleItem(circ)
            self.addItem(roi)
            self._listenRoi(i, roi)
            self._rois.append(roi)

    def _listenRoi(self, i: int, roi: QGraphicsEllipseItem):
        roi.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: int):
        print(f'Selected {i}')
        if i != self._selected:
            if not i is None:
                self._rois[i].select()
            if not self._selected is None:
                self._rois[self._selected].unselect()
            self._selected = i

    def _delete(self):
        print('delete')
        if self._editable and not self._selected is None:
            self._circles.pop(self._selected)
            r = self._rois.pop(self._selected)
            self.removeItem(r)
            self._selected = None
            self.edited.emit()

    def _edit(self):
        print('edit')
        if self._editable and not self._is_drawing:
            self._select(None)
            self._is_drawing = True
            assert not self._next_label is None, 'No next label'
            print('Next label:', self._next_label)
            self._drawn_lbl = self._next_label
            self._vb = self.getViewBox()
            self._vb.setMouseEnabled(x=False, y=False)

    def _escape(self):
        print('escape')
        if not self._drawn_roi is None:
            self.removeItem(self._drawn_roi)
        self._reset_drawing_state()
        self._select(None)

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()]) 
                if self._drawn_pos is None:
                    print('starting draw')
                    self._drawn_pos = pos.copy() 
                else:
                    r = np.linalg.norm(pos - self._drawn_pos)
                    if r > 0:
                        # print(f'Drawing radius: {r}')
                        if self._drawn_roi is None:
                            # print('Made roi')
                            self._drawn_roi = LabeledCircleItem(LabeledCircle(self._drawn_lbl, self._drawn_pos.copy(), r))
                        else:
                            # print('Editing roi')
                            if not self._roi_added:
                                self.addItem(self._drawn_roi) # Workaround for snapping bug
                                self._roi_added = True
                            self._drawn_roi.setRadius(r)
            else:
                if not self._drawn_roi is None:
                    print('ending draw')
                    self._circles.append(self._drawn_roi._circ)
                    self._rois.append(self._drawn_roi)
                    N = len(self._circles)-1
                    self._listenRoi(N, self._drawn_roi)
                    self._reset_drawing_state()
                    self._select(N)
                    self.edited.emit()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._drawn_lbl = None
        self._drawn_pos = None
        self._drawn_roi = None
        self._roi_added = False
        if hasattr(self, '_vb') and not self._vb is None:
            self._vb.setMouseEnabled(x=True, y=True)
        self._vb = None
