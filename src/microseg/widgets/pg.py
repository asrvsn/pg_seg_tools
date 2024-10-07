'''
Pyqtgraph widgets
'''
import os
from typing import Tuple, Optional, List, Callable
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QShortcut
import pickle

import microseg.utils.mask as mutil
from microseg.utils.colors import *
from microseg.utils.data import *
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

class ImageItem(pg.ImageItem):
    '''
    ImageItem correcting the insane behavior of pyqtgraph with respect to orientation
    '''
    def setImage(self, img: np.ndarray):
        img = np.swapaxes(img, 0, 1)
        img = img[:, ::-1, ...]
        super().setImage(img)

class ImagePlotWidget(NoTouchPlotWidget):
    '''
    Plot widget with default image item, zoom limits set to image
    '''
    def __init__(self, *args, limit_zoom: bool=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAspectLocked(True)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self._img = ImageItem()
        self.addItem(self._img)
        if limit_zoom:
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

class MaskItem(ImageItem):
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
                    self._drawn_poly = [x, y]
                else:
                    self._drawn_poly.append(x)
                    self._drawn_poly.append(y)
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
        self._img_item = ImageItem()
        self.addItem(self._img_item)
        self._mask_item = EditableMaskItem() if editable else MaskItem()
        self._mask_item.add_to_plot(self)

        # State
        self._img = None
        self._mask = None

    def setData(self, img: np.ndarray, mask: np.ndarray):
        assert img.shape[:2] == mask.shape, 'Image and mask must have same shape'
        self._img = img
        self._mask = mask
        self._img_item.setImage(img)
        self._mask_item.setMask(mask)

class ThingsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for drawing things on an image
    '''
    edited = QtCore.Signal()
    undo_n: int=100

    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable

        # State
        self._things: List[LabeledThing] = []
        self._items: List[SelectableItem] = []
        self._selected: int = None
        self._shortcuts = []
        self._vb = None
        self._reset_drawing_state()
        self._next_label: int = None
        self._undo_stack = None # Uninitialized
        self._redo_stack = None

        if editable:
            self.add_sc('Delete', lambda: self._delete())
            self.add_sc('Backspace', lambda: self._delete())
            self.add_sc('E', lambda: self._edit())
            self.add_sc('Ctrl+Z', lambda: self._undo())
            self.add_sc('Ctrl+Y', lambda: self._redo())
            self.add_sc('Ctrl+Shift+Z', lambda: self._redo())
        self.add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    def add_sc(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)

    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

    @abc.abstractmethod
    def createItemFromThing(self, thing: LabeledThing) -> SelectableItem:
        pass

    @abc.abstractmethod
    def getThingFromItem(self, item: SelectableItem) -> LabeledThing:
        pass

    def setThings(self, things: List[LabeledThing], reset_stacks: bool=True):
        # Remove previous things
        self._selected = None
        self._things = things
        for item in self._items:
            self.removeItem(item)
        # Add new things
        self._items = []
        for i, thing in enumerate(things):
            item = self.createItemFromThing(thing)
            self.addItem(item)
            self._listenItem(i, item)
            self._items.append(item)
        if reset_stacks:
            self._undo_stack = [self.getThings()]
            self._redo_stack = []
        self._reset_drawing_state()

    def getThings(self) -> List[LabeledThing]:
        return [t.copy() for t in self._things]

    def _listenItem(self, i: int, item: SelectableItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: int):
        print(f'Selecting {i}')
        if i != self._selected:
            if not i is None:
                self._items[i].select()
            if not self._selected is None:
                self._items[self._selected].unselect()
            self._selected = i

    def _delete(self):
        print('delete')
        if self._editable and not self._selected is None:
            self._things.pop(self._selected)
            item = self._items.pop(self._selected)
            self.removeItem(item)
            self._selected = None
            self._push_stack()

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
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._reset_drawing_state()
        self._select(None)

    @abc.abstractmethod
    def initDrawingState(self):
        ''' Reset the drawing state specific to this item '''
        pass

    def _reset_drawing_state(self):
        self.initDrawingState()
        self._is_drawing = False
        self._drawn_lbl = None
        self._drawn_item = None
        if not self._vb is None:
            self._vb.setMouseEnabled(x=True, y=True)
        self._vb = None

    @abc.abstractmethod
    def modifyDrawingState(self, pos):
        ''' Modify the drawing state from given pos specific to this item '''
        pass

    @abc.abstractmethod
    def finishDrawingState(self):
        ''' Finish the drawing state specific to this item '''
        pass

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()])
                self.modifyDrawingState(pos)
            else:
                if not self._drawn_item is None:
                    print('ending draw')
                    self.finishDrawingState()
                    thing = self.getThingFromItem(self._drawn_item)
                    self._things.append(thing)
                    self._items.append(self._drawn_item)
                    N = len(self._things)-1
                    self._listenItem(N, self._drawn_item)
                    self._reset_drawing_state()
                    self._push_stack()
                    # self._select(N)

    def _push_stack(self):
        self._undo_stack.append(self.getThings())
        self._undo_stack = self._undo_stack[-self.undo_n:]
        self._redo_stack = []
        print(f'Current stacks: undo {len(self._undo_stack)}, redo {len(self._redo_stack)}')
        self.edited.emit()

    def _undo(self):
        print('undo')
        if len(self._undo_stack) > 1:
            self._redo_stack.append(self._undo_stack[-1])
            self._undo_stack = self._undo_stack[:-1]
            things = [t.copy() for t in self._undo_stack[-1]]
            self.setThings(things, reset_stacks=False)
            self.edited.emit()
        else:
            print('Cannot undo further')

    def _redo(self):
        print('redo')
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self._redo_stack[-1])
            self._redo_stack = self._redo_stack[:-1]
            things = [t.copy() for t in self._undo_stack[-1]]
            self.setThings(things, reset_stacks=False)
            self.edited.emit()
        else:
            print('Cannot redo further')

class CirclesImageWidget(ThingsImageWidget):
    '''
    Editable widget for drawing circles on an image
    '''
    def __init__(self, *args, **kwargs):
        self._usePoly: bool = True
        super().__init__(*args, **kwargs)
        self.add_sc('P', lambda: self._toggle_use_poly())

    def _toggle_use_poly(self):
        self._usePoly = not self._usePoly
        print(f'Using poly: {self._usePoly}')
    
    def createItemFromThing(self, circ: LabeledCircle) -> SelectableCircleItem:
        return SelectableCircleItem(circ)
    
    def getThingFromItem(self, item: SelectableCircleItem) -> LabeledCircle:
        return item._circ

    def initDrawingState(self):
        self._drawn_pos = None
        self._item_added = False
        PolysImageWidget.initDrawingState(self)

    def modifyDrawingState(self, pos: np.ndarray):
        if self._usePoly:
            PolysImageWidget.modifyDrawingState(self, pos)
        else:
            if self._drawn_pos is None:
                print('starting draw')
                self._drawn_pos = pos.copy() 
            else:
                r = np.linalg.norm(pos - self._drawn_pos)
                if r > 0:
                    # print(f'Drawing radius: {r}')
                    if self._drawn_item is None:
                        # print('Made roi')
                        self._drawn_item = self.createItemFromThing(LabeledCircle(self._drawn_lbl, self._drawn_pos.copy(), r))
                    else:
                        # print('Editing roi')
                        if not self._item_added:
                            self.addItem(self._drawn_item) # Workaround for snapping bug
                            self._item_added = True
                        self._drawn_item.setRadius(r)

    def finishDrawingState(self):
        if self._usePoly:
            poly = PolysImageWidget.getThingFromItem(self, self._drawn_item)
            circ = Sphere.from_poly(poly)
            circ = LabeledCircle(poly.l, circ.v, circ.r)
            self.removeItem(self._drawn_item)
            self._drawn_item = self.createItemFromThing(circ)
            self.addItem(self._drawn_item)

class PolysImageWidget(ThingsImageWidget):
    '''
    Editable widget for drawing polygons on an image
    '''
    def __init__(self, *args, **kwargs):
        self._hullify: bool = True
        super().__init__(*args, **kwargs)
        self.add_sc('H', lambda: self._toggle_hullify())

    def _toggle_hullify(self):
        self._hullify = not self._hullify
        print(f'Hullify: {self._hullify}')
    
    def createItemFromThing(self, poly: LabeledPolygon) -> SelectablePolygonItem:
        return SelectablePolygonItem(poly)
    
    def getThingFromItem(self, item: SelectablePolygonItem) -> LabeledPolygon:
        return item._poly
    
    def initDrawingState(self):
        self._drawn_poses = []

    def modifyDrawingState(self, pos: np.ndarray):
        self._drawn_poses.append(pos.copy())
        vertices = np.array(self._drawn_poses)
        if vertices.shape[0] > 2:
            if not self._drawn_item is None:
                self.removeItem(self._drawn_item)
            poly = LabeledPolygon(self._drawn_lbl, vertices, use_chull_if_invalid=True)
            self._drawn_item = SelectablePolygonItem(poly)
            self.addItem(self._drawn_item)

    def finishDrawingState(self):
        if self._hullify:
            poly = self.getThingFromItem(self._drawn_item)
            poly = LabeledPolygon.from_pointcloud(poly.l, poly.vertices)
            self.removeItem(self._drawn_item)
            self._drawn_item = self.createItemFromThing(poly)
            self.addItem(self._drawn_item)

class ThingSegmentorWidget(SaveableWidget, metaclass=QtABCMeta):
    @abc.abstractmethod
    def makeWidget(*args, **kwargs) -> ThingsImageWidget:
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create the widget
        self._editor = self.makeWidget(limit_zoom=False)
        self._main_layout.addWidget(self._editor)
        self._nthings = QLabel()
        self._settings_layout.addWidget(self._nthings)
        self._z_slider = IntegerSlider(mode='scroll')
        self._settings_layout.addWidget(self._z_slider)

        # Listeners
        self._editor.edited.connect(self._edited)
        self._z_slider.valueChanged.connect(lambda z: self._setZ(z, set_slider=False))

        # State
        self._z = 0
        self._img: np.ndarray = None
        self._things: List[LabeledThing] = []

    def setData(self, img: np.ndarray, things: List[LabeledThing]=[]):
        self._img = img
        self._things = things
        self._setZ(0)
        self._editor.setThings(things)
        self._advance_label()
        self._nthings.setText(f'Things: {len(things)}')
        self.setDisabled(False)

    def _setZ(self, z: int, set_slider: bool=True):
        ''' Determines the z-axis behavior and view '''
        if self._img.ndim == 2 or (self._img.ndim in [3, 4] and self._img.shape[0] == 1):
            self._z = 0
            img = self._img if self._img.ndim == 2 else self._img[0]
            if set_slider:
                print(f'Received standard 2D image, disabling z-slider')
                self._z_slider.hide()
        else:
            self._z = z
            img = self._img[z]
            if set_slider:
                print(f'Received z-stack of shape {self._img.shape}, enabling z-slider')
                self._z_slider.show()
                self._z_slider.setData(0, self._img.shape[0]-1, z)
        self._editor.setImage(img)

    def _advance_label(self):
        l = 0
        for t in self._things:
            l = max(l, t.l)
        l += 1
        self._editor._next_label = l

    def _edited(self):
        self._things = self._editor.getThings()
        self._advance_label()
        self._nthings.setText(f'Things: {len(self._things)}')
        print('things edited')

    def getData(self) -> List[LabeledThing]:
        return self._things
    
class ThingsSegmentorWindow(MainWindow):
    def __init__(self, path: str, segmentor: ThingSegmentorWidget, chan: Optional[int]=None, desc: str='things', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        self._descriptor = desc
        img = load_stack(path) # ZXYC
        print(f'Loaded image from {path} with shape {img.shape}')
        if chan is None:
            assert img.shape[-1] in [1, 3], f'Cannot interpret as grayscale or rgb, pass a channel index'
            print(f'No channel specified; interpreting multichannel image as grayscale or rgb')
        else:
            img = img[:, :, :, chan]
        self._ylen = img.shape[2]
        self._things_path = f'{os.path.splitext(path)[0]}.{desc}'

        # Load existing things if exists
        things = []
        if os.path.isfile(self._things_path):
            print(f'Loading {desc} from {self._things_path}')
            things = pickle.load(open(self._things_path, 'rb'))
        # Account for pyqtgraph orientation
        things = [t.flipy(self._ylen) for t in things]

        pg.setConfigOptions(antialias=True, useOpenGL=False)
        self.setWindowTitle(f'{desc} segmentor')
        self._seg = segmentor
        self.setCentralWidget(segmentor)
        self._seg.setData(img, things=things)
        self.resizeToActiveScreen()

        # Listeners
        self._seg.saved.connect(self._save)

    def _save(self):
        things = self._seg.getData()
        # Account for pyqtgraph orientation
        things = [t.flipy(self._ylen) for t in things]
        pickle.dump(things, open(self._things_path, 'wb'))
        print(f'Saved {self._descriptor} to {self._things_path}')