'''
Base classes for building apps from ROI editors
'''

from .base import *
from .pg import *
from .roi_image import *

class ROIsSingleImageApp(SaveableWidget):
    '''
    Abstract base app for editing ROIs on a single image with undo/redo stack
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._editor = ROIsImageWidget(editable=True)
        self._main_layout.addWidget(self._editor)
        self._count_lbl = QLabel()
        self._settings_layout.addWidget(self._count_lbl)

        # Listeners
        self._editor.proposeAdd.connect(self.add)
        self._editor.proposeDelete.connect(self.delete)
        self._editor.proposeUndo.connect(self.undo)
        self._editor.proposeRedo.connect(self.redo)

        # State (override)

    @abc.abstractmethod
    def add(self, polys: List[PlanarPolygon]):
        pass

    @abc.abstractmethod
    def delete(self, indices: List[int]):
        pass

    @abc.abstractmethod
    def undo(self):
        pass

    @abc.abstractmethod
    def redo(self):
        pass