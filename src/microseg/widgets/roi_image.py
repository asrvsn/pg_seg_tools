'''
Image + ROI overlays
- Support for manually drawing polygons
- Support for semi-automatically drawing polygons via standard segmentation algorithms (cellpose, watershed, etc.)
'''
from typing import Set
from qtpy.QtWidgets import QRadioButton, QLabel, QCheckBox, QComboBox

from .pg import *
from .roi import *

class ROIsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for displaying and drawing ROIs on an image
    '''
    proposeAdd = QtCore.Signal(List[PlanarPolygon]) # Add polygons
    proposeDelete = QtCore.Signal(Set[int]) # Delete by label
    proposeUndo = QtCore.Signal()
    proposeRedo = QtCore.Signal()

    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable

        # State
        self._rois: List[LabeledROI] = []
        self._items: List[LabeledROIItem] = []
        self._selected = []
        self._shortcuts = []
        self._vb = None
        self._reset_drawing_state()

        # Listeners
        if editable:
            self.add_sc('Delete', lambda: self._delete())
            self.add_sc('Backspace', lambda: self._delete())
            self.add_sc('E', lambda: self._edit())
            self.add_sc('Ctrl+Z', lambda: self.proposeUndo.emit())
            self.add_sc('Ctrl+Y', lambda: self.proposeRedo.emit())
            self.add_sc('Ctrl+Shift+Z', lambda: self.proposeRedo.emit())
        self.add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    def add_sc(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self.setImage(img)
        self.setROIs(rois)

    def setImage(self, img: np.ndarray):
        self._img.setImage(img)

    def setROIs(self, rois: List[LabeledROI]):
        # Remove previous rois
        self._selected = []
        self._rois = rois
        for item in self._items:
            self.removeItem(item)
        # Add new rois
        self._items = []
        for i, roi in enumerate(rois):
            item = roi.to_item(self.shape)
            self.addItem(item)
            self._listenItem(i, item)
            self._items.append(item)
        self._reset_drawing_state()

    def getROIs(self) -> List[LabeledROI]:
        return [r.copy() for r in self._rois]
    
    def _listenItem(self, i: int, item: LabeledROIItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: Optional[int]):
        if i is None:
            self._unselect_all()
        if not i in self._selected:
            if not QGuiApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._unselect_all()
            self._selected.append(i)
            self._items[i].select()
        print(f'Selected: {self._selected}')

    def _unselect_all(self):
        for i in self._selected:
            self._items[i].unselect()
        self._selected = []

    def _delete(self):
        if self._editable and len(self._selected) > 0:
            print(f'propose delete {len(self._selected)} things')
            self.proposeDelete.emit(set(self._selected))

    def _edit(self):
        print('edit')
        if self._editable and not self._is_drawing:
            self._select(None)
            self._is_drawing = True
            self._vb = self.getViewBox()
            self._vb.setMouseEnabled(x=False, y=False)

    def _escape(self):
        print('escape')
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._reset_drawing_state()
        self._select(None)

    def _init_drawing_state(self):
        self._drawn_poses = []

    def _reset_drawing_state(self):
        self._init_drawing_state()
        self._is_drawing = False
        if not self._drawn_item is None:
            self.removeItem(self._drawn_item)
        self._drawn_item = None
        if not self._vb is None:
            self._vb.setMouseEnabled(x=True, y=True)
        self._vb = None

    def _modify_drawing_state(self, pos: np.ndarray):
        self._drawn_poses.append(pos.copy())
        vertices = np.array(self._drawn_poses)
        if vertices.shape[0] > 2:
            if not self._drawn_item is None:
                self.removeItem(self._drawn_item)
            lroi = LabeledROI(65, PlanarPolygon(vertices, use_chull_if_invalid=True)).fromPyQTOrientation(self.shape)
            self._drawn_item = lroi.to_item(self.shape) # Original vertices already in PyQT orientation, do the identity transform
            self.addItem(self._drawn_item)

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()])
                self.modifyDrawingState(pos)
            else:
                if not self._drawn_item is None:
                    print('ending draw')
                    poly = self._drawn_item.to_roi(self.shape).roi
                    self._reset_drawing_state()
                    print('propose add 1 poly manually')
                    self.proposeAdd.emit([poly])

class ROIsCreator(PaneledWidget):
    '''
    Thin wrapper around ROIsImageWidget for creating ROIs with several options
    Expects image in XYC format
    '''
    proposeAdd = QtCore.Signal(List[ROI])
    proposeDelete = QtCore.Signal(Set[int])
    AVAIL_MODES = [
        ('Polygon', lambda self, polys: [
            p.hullify() if self._use_chull else p for p in polys
        ]),
        ('Ellipse', lambda self, polys: [
            Ellipse.from_poly(p) for p in polys
        ]),
        ('Circle', lambda self, polys: [
            Circle.from_poly(p) for p in polys
        ]),
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._image = ROIsImageWidget(editable=True)
        self._main_layout.addWidget(self._image)
        self._settings_layout.addWidget(QLabel('Mode:'))
        self._mode_btns = []
        for i in range(len(self.AVAIL_MODES)):
            self._add_mode(i)
        self._chull_box = QCheckBox('Convex hull')
        self._settings_layout.addWidget(self._chull_box)
        self._rgb_box = QCheckBox('Interpret RGB')
        self._settings_layout.addWidget(self._rgb_box)
        self._rgb_box.hide()
        self._chan_slider = IntegerSlider(mode='scroll')
        self._settings_layout.addWidget(self._chan_slider)
        self._chan_slider.hide()
        self._count_lbl = QLabel()
        self._settings_layout.addWidget(self._count_lbl)

        # Listeners
        self._chull_box.stateChanged.connect(lambda _: self._update_settings(redraw=False))
        self._rgb_box.stateChanged.connect(lambda _: self._update_settings(redraw=True))
        self._chan_slider.valueChanged.connect(lambda _: self._update_settings(redraw=True))
        # Bubble up other signals
        self._image.proposeAdd.connect(lambda polys: self.proposeAdd.emit(self._make_rois(polys)))
        self._image.proposeDelete.connect(self.proposeDelete.emit)

        # State
        self._mode = 0
        self._mode_btns[self._mode].setChecked(True)
        self._chull_box.setChecked(True)
        self._rgb_box.setChecked(True)
        self._chan_slider.setData(0, 255, 0)
        self._update_settings(redraw=False)
        self._img = None

    def _add_mode(self, i: int):
        mode, _ = self.AVAIL_MODES[i]
        btn = QRadioButton(mode)
        self._mode_btns.append(btn)
        self._settings_layout.addWidget(btn)
        btn.clicked.connect(lambda: self._set_mode(i))

    def _set_mode(self, i: int):
        self._mode = i
        if self.AVAIL_MODES[i][0] == 'Polygon':
            self._chull_box.show()
        else:
            self._chull_box.hide()

    def _update_settings(self, redraw: bool=True):
        self._use_chull = self._chull_box.isChecked()
        self._interpret_rgb = self._rgb_box.isChecked()
        self._chan = self._chan_slider.value()
        if redraw:
            self.setImage(self._img)

    def _set_count(self, n: int):
        self._count_lbl.setText(f'Objects: {n}')
        
    # @property
    # def next_label(self) -> int:
    #     return 0 if len(self._rois) == 0 else (max(r.l for r in self._rois) + 1)
        
    def _make_rois(self, polys: List[PlanarPolygon]):
        return self.AVAIL_MODES[self._mode][1](self, polys)
        # lbl = self.next_label
        # new_rois = [LabeledROI(i+lbl, r) for i, r in enumerate(new_rois)]
        # self._rois.extend(new_rois)
        # self._image.setROIs(self._rois)

    # def delete(self, labels: Set[int]):
    #     self._rois = [r for r in self._rois if not (r.l in labels)]
    #     self._image.setROIs(self._rois)

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self._image.setData(img, rois)

    def setImage(self, img: np.ndarray):
        assert img.ndim in [2, 3], 'Expected XY or XYC image'
        if img.ndim == 2 or img.shape[2] == 1:
            self._rgb_box.hide()
            self._chan_slider.hide()
            img = img if img.ndim == 2 else img[:, :, 0]
            self._image.setImage(img)
        else:
            if img.shape[2] == 3:
                self._rgb_box.show()
                self._rgb_box.setChecked(self._interpret_rgb)
            if self._interpret_rgb:
                self._chan_slider.hide()
                self._image.setImage(img)
            else:
                self._chan_slider.show()
                cmax = img.shape[2] - 1
                self._chan = min(self._chan, cmax)
                self._chan_slider.setData(0, cmax, self._chan)
                self._image.setImage(img[:, :, self._chan])

    def setROIs(self, rois: List[LabeledROI]):
        self._image.setROIs(rois)
        self._set_count(len(rois))

class OldWidget:
    edited = QtCore.Signal()
    undo_n: int=100

    def __init__(self, editable: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._editable = editable

        # State
        self._rois: List[LabeledROI] = []
        self._items: List[LabeledROIItem] = []
        self._selected = []
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
    def createItemFromThing(self, thing: LabeledROI) -> LabeledROIItem:
        pass

    @abc.abstractmethod
    def getThingFromItem(self, item: LabeledROIItem) -> LabeledROI:
        pass

    def setThings(self, things: List[LabeledROI], reset_stacks: bool=True):
        # Remove previous things
        self._selected = []
        self._rois = things
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

    def getThings(self) -> List[LabeledROI]:
        return [t.copy() for t in self._rois]

    def _listenItem(self, i: int, item: LabeledROIItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: Optional[int]):
        if i is None:
            self._unselect_all()
        if not i in self._selected:
            if not QGuiApplication.keyboardModifiers() & Qt.ShiftModifier:
                self._unselect_all()
            self._selected.append(i)
            self._items[i].select()
        print(f'Selected: {self._selected}')

    def _unselect_all(self):
        for i in self._selected:
            self._items[i].unselect()
        self._selected = []

    def _delete(self):
        if self._editable and len(self._selected) > 0:
            print(f'delete {len(self._selected)} things')
            for i in self._selected:
                self.removeItem(self._items[i])
            selset = set(self._selected)
            self._rois = [t for i, t in enumerate(self._rois) if not i in selset]
            self._items = [t for i, t in enumerate(self._items) if not i in selset]
            self._selected = []
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
                    self._rois.append(thing)
                    self._items.append(self._drawn_item)
                    N = len(self._rois)-1
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