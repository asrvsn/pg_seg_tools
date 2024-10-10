'''
Base classes for building apps from ROI editors
'''
from qtpy.QtWidgets import QRadioButton, QLabel, QCheckBox, QComboBox

from .base import *
from .pg import *
from .roi_image import *

class ROIsSingleImageApp(SaveableWidget):
    '''
    Abstract base app for editing ROIs on a single image with undo/redo stack
    '''
    undo_n = 100
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
        self._editor = ROIsImageWidget(editable=True)
        self._main_layout.addWidget(self._editor)
        self._settings_layout.addWidget(QLabel('Mode:'))
        self._mode_btns = []
        for i in range(len(self.AVAIL_MODES)):
            self._add_mode(i)
        self._use_chull_box = QCheckBox('Convex hull')
        self._settings_layout.addWidget(self._use_chull_box)

        self._count_lbl = QLabel()
        self._settings_layout.addWidget(self._count_lbl)

        # Listeners
        self._editor.proposeAdd.connect(self.add)
        self._editor.proposeDelete.connect(self.delete)
        self._editor.proposeUndo.connect(self.undo)
        self._editor.proposeRedo.connect(self.redo)

        # State
        self._use_chull = True
        self._use_chull_box.setChecked(self._use_chull)
        self._mode = 0
        self._mode_btns[self._mode].setChecked(True)
        self._rois = []

    def _add_mode(self, i: int):
        mode, _ = self.AVAIL_MODES[i]
        btn = QRadioButton(mode)
        self._mode_btns.append(btn)
        self._settings_layout.addWidget(btn)
        btn.clicked.connect(lambda: self._set_mode(i))

    def _set_mode(self, i: int):
        self._mode = i
        if self.AVAIL_MODES[i][0] == 'Polygon':
            self._use_chull_box.show()
        else:
            self._use_chull_box.hide()
        
    @property
    def next_label(self) -> int:
        return 0 if len(self._rois) == 0 else (max(r.l for r in self._rois) + 1)
        
    def add(self, polys: List[PlanarPolygon]):
        new_rois = self.AVAIL_MODES[self._mode][1](self, polys)
        lbl = self.next_label
        new_rois = [LabeledROI(i+lbl, r) for i, r in enumerate(new_rois)]
        self._rois.extend(new_rois)
        self._editor.setROIs(self._rois)

    def delete(self, labels: Set[int]):
        self._rois = [r for r in self._rois if not (r.l in labels)]
        self._editor.setROIs(self._rois)

    @abc.abstractmethod
    def undo(self):
        pass

    @abc.abstractmethod
    def redo(self):
        pass