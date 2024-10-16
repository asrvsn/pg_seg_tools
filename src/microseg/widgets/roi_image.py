'''
Image + ROI overlays
- Support for manually drawing polygons
- Support for semi-automatically drawing polygons via standard segmentation algorithms (cellpose, watershed, etc.)
'''
from typing import Set, Type
from qtpy.QtWidgets import QRadioButton, QLabel, QCheckBox, QComboBox

from .pg import *
from .roi import *
from .seg import *

class ROIsImageWidget(ImagePlotWidget, metaclass=QtABCMeta):
    '''
    Editable widget for displaying and drawing ROIs on an image
    '''
    add = QtCore.Signal(object) # PlanarPolygon
    delete = QtCore.Signal(object) # Set[int], indices into ROIs

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
            self._add_sc('Delete', lambda: self._delete())
            self._add_sc('Backspace', lambda: self._delete())
            self._add_sc('E', lambda: self._edit())
        self._add_sc('Escape', lambda: self._escape())
        self.scene().sigMouseMoved.connect(self._mouse_move)

    ''' API methods '''

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
            item = roi.toItem(self._shape())
            self.addItem(item)
            self._listen_item(i, item)
            self._items.append(item)
        self._reset_drawing_state()

    ''' Private methods '''

    def _add_sc(self, key: str, fun: Callable):
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(fun)
        self._shortcuts.append(sc)

    def _listen_item(self, i: int, item: LabeledROIItem):
        item.sigClicked.connect(lambda: self._select(i))

    def _select(self, i: Optional[int]):
        if i is None:
            self._unselect_all()
        elif not i in self._selected:
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
            self.delete.emit(set(self._selected))

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
            lroi = LabeledROI(65, PlanarPolygon(vertices, use_chull_if_invalid=True)).fromPyQTOrientation(self._shape())
            self._drawn_item = lroi.toItem(self._shape()) # Original vertices already in PyQT orientation, do the identity transform
            self.addItem(self._drawn_item)

    def _mouse_move(self, pos):
        if self._is_drawing:
            if QtCore.Qt.LeftButton & QtWidgets.QApplication.mouseButtons():
                pos = self._vb.mapSceneToView(pos)
                pos = np.array([pos.x(), pos.y()])
                self._modify_drawing_state(pos)
            else:
                if not self._drawn_item is None:
                    print('ending draw')
                    poly = self._drawn_item.toROI(self._shape()).roi
                    self.removeItem(self._drawn_item)
                    self._reset_drawing_state()
                    self.add.emit(poly)

class ROIsCreator(PaneledWidget):
    '''
    Thin wrapper around ROIsImageWidget for creating ROIs with several options
    - convert drawn poly to ellipse or circle
    - take convex hull of drawn poly
    - channel selector for image
    Expects image in XYC format
    '''
    add = QtCore.Signal(object) # List[ROI]
    delete = QtCore.Signal(object) # Set[int], labels of deleted ROIs
    AVAIL_MODES: List[SegmentorWidget] = [
        ManualSegmentorWidget,
        # CellposeSegmentorWidget,
    ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._widget = ROIsImageWidget(editable=True)
        self._main_layout.addWidget(self._widget)
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(QLabel('Mode:'))
        self._bottom_layout.addSpacing(10)
        self._segmentors = [self._add_mode(c) for c in self.AVAIL_MODES]
        self._mode_drop = QComboBox()
        self._mode_drop.addItems([s.name() for s in self._segmentors])
        self._bottom_layout.addWidget(self._mode_drop)
        self._bottom_layout.addSpacing(10)
        self._options_box = QCheckBox('Show options')
        self._bottom_layout.addWidget(self._options_box)
        self._bottom_layout.addStretch()
        self._bottom_layout.addWidget(QLabel('Chan:'))
        self._chan_slider = IntegerSlider(mode='scroll')
        self._bottom_layout.addWidget(self._chan_slider)
        self._chan_slider.hide()
        self._rgb_box = QCheckBox('Interpret RGB')
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._rgb_box)
        self._rgb_box.hide()
        self._count_lbl = QLabel()
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._count_lbl)
        self._proposals_box = QCheckBox('Proposals only')
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._proposals_box)
        self._proposals_lbl = QLabel()
        self._bottom_layout.addSpacing(10)
        self._bottom_layout.addWidget(self._proposals_lbl)

        # State
        self._mode = 0
        self._img = None
        self._rois = []
        self._set_proposing(False)
        self._proposed_rois = []
        self._rgb_box.setChecked(True)
        self._options_box.setChecked(True)
        self._chan_slider.setData(0, 255, 0)
        self._proposals_box.setChecked(False)
        self._update_settings(redraw=False)

        # Listeners
        self._mode_drop.currentIndexChanged.connect(self._set_mode)
        self._options_box.stateChanged.connect(lambda _: self._update_settings(redraw=False))
        self._rgb_box.stateChanged.connect(lambda _: self._update_settings(redraw=True))
        self._chan_slider.valueChanged.connect(lambda _: self._update_settings(redraw=True))
        self._proposals_box.stateChanged.connect(lambda _: self._set_only_proposals())
        self._widget.add.connect(self._add_from_child)
        self._widget.delete.connect(self._delete_from_child)
        options_sc = QShortcut(QKeySequence('O'), self)
        options_sc.activated.connect(lambda: self._options_box.setChecked(not self._options_box.isChecked()))

    ''' API methods '''

    def setData(self, img: np.ndarray, rois: List[LabeledROI]):
        self.setImage(img)
        self.setROIs(rois)

    def setImage(self, img: np.ndarray):
        assert img.ndim in [2, 3], 'Expected XY or XYC image'
        self._img = img
        if img.ndim == 2 or img.shape[2] == 1:
            self._rgb_box.hide()
            self._chan_slider.hide()
            img = img if img.ndim == 2 else img[:, :, 0]
            self._widget.setImage(img)
        else:
            if img.shape[2] == 3:
                self._rgb_box.show()
                self._rgb_box.setChecked(self._interpret_rgb)
            if self._interpret_rgb:
                self._chan_slider.hide()
                self._widget.setImage(img)
            else:
                self._chan_slider.show()
                cmax = img.shape[2] - 1
                self._chan = min(self._chan, cmax)
                self._chan_slider.setData(0, cmax, self._chan)
                self._widget.setImage(img[:, :, self._chan])

    def setROIs(self, rois: List[LabeledROI]):
        assert not self._is_proposing
        self._rois = rois
        self._widget.setROIs(rois)
        self._count_lbl.setText(f'Current: {len(rois)}')

    ''' Private methods '''

    def _add_mode(self, Cls: Type[SegmentorWidget]) -> SegmentorWidget:
        seg = Cls(self)
        seg.propose.connect(self._propose)
        seg.add.connect(self._add)
        seg.cancel.connect(self._cancel)
        return seg

    def _set_mode(self, i: int):
        self._mode = i

    def _update_settings(self, redraw: bool=True):
        self._show_options = self._options_box.isChecked()
        self._interpret_rgb = self._rgb_box.isChecked()
        self._chan = self._chan_slider.value()
        self._only_proposals = self._proposals_box.isChecked()
        if redraw:
            self.setImage(self._img)

    def _set_only_proposals(self):
        assert self._is_proposing
        self._update_settings(redraw=False)
        self._propose(self._proposed_rois)

    def _add_from_child(self, poly: PlanarPolygon):
        '''
        Process add() signal from child
        '''
        assert not self._is_proposing
        assert len(self._proposed_rois) == 0
        self._is_proposing = True
        if self._show_options:
            self._segmentors[self._mode].prompt(poly)
        else:
            self._segmentors[self._mode].prompt_immediate(poly)

    def _delete_from_child(self, indices: Set[int]):
        '''
        Process delete() signal from child. 
        '''
        if self._is_proposing: 
            # Allow deletions from proposals only, shift index accordingly
            n = len(self._rois)
            proposals = [r for i, r in enumerate(self._proposed_rois) if not (i + n in indices)]
            self._propose(proposals)
        else: 
            # Convert indices to labels and bubble to parent
            lbls = {self._rois[i].lbl for i in indices}
            self.delete.emit(lbls) 

    def _propose(self, rois: List[ROI]):
        '''
        Show additional proposed ROIs at this level without bubbling them to the parent yet.
        Intended to be called any number of times in sequence.
        '''
        if not self._is_proposing:
            self._set_proposing(True)
        self._proposed_rois = rois 
        proposed_labeled = [LabeledROI(65, r) for r in rois] # All with that color
        if self._only_proposals:
            self._widget.setROIs(proposed_labeled) 
        else:
            self._widget.setROIs(self._rois + proposed_labeled)

    def _add(self):
        '''
        Process add() signal from segmentor, which asynchronously bubbles add() signal from child.
        '''
        assert self._is_proposing
        self._set_proposing(False)
        if len(self._proposed_rois) > 0:
            self.add.emit(self._proposed_rois)
        self._proposed_rois = []

    def _cancel(self):
        '''
        Process cancel() signal from segmentor, does not fire add() signal to parent.
        '''
        assert self._is_proposing
        self._set_proposing(False)
        self._proposed_rois = []
        self._widget.setROIs(self._rois)

    def _set_proposing(self, bit: bool):
        self._is_proposing = bit
        if bit:
            self._proposals_box.show()
            self._proposals_lbl.show()
            self._mode_drop.setEnabled(False)
        else:
            self._proposals_box.hide()
            self._proposals_lbl.hide()
            self._mode_drop.setEnabled(True)