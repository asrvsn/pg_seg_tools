'''
Widget to register points in 3d
'''
import os
import numpy as np
import pyqtgraph.opengl as gl
from scipy.spatial.distance import pdist

from matgeo import Triangulation

from .widgets.base import *
from .widgets.pg_gl import *
from .widgets.seg_2d import *
from .utils.colors import map_colors

class Register3DWidget(SaveableWidget):
    undo_n: int = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._vw = GLSelectableSurfaceViewWidget(show_normals=True)
        self._main_layout.addWidget(self._vw)
        self._dist_lbl = QLabel(f'Distance: -')
        self._settings_layout.addWidget(self._dist_lbl)

        # Listeners
        self._c_sc = QShortcut(QKeySequence('C'), self._vw)
        self._c_sc.activated.connect(self._collapse_labels)
        self._u_sc = QShortcut(QKeySequence('Ctrl+Z'), self._vw)
        self._u_sc.activated.connect(self._undo)
        self._r_sc = QShortcut(QKeySequence('Ctrl+Shift+Z'), self._vw)
        self._r_sc.activated.connect(self._redo)
        self._vw.selectionChanged.connect(self._selection_changed)

        # State
        self._tri = None
        self._undo_stack = []
        self._redo_stack = []

    def setData(self, pts: np.ndarray):
        self.setDisabled(False)
        self._redraw(pts)
        self._undo_stack.append(self._tri.copy())
        self._redo_stack = []

    def getData(self) -> np.ndarray:
        return self._tri.pts.copy()

    def _redraw(self, pts: np.ndarray):
        self._tri = Triangulation.surface_3d(pts, method='advancing_front')
        self._vw.setMeshData(self._tri)

    def _collapse_labels(self):
        if not (self._tri is None):
            sel = list(self._vw._selected)
            if len(sel) > 1:
                print(f'Collapsing {sel}')
                pts_unsel = np.delete(self._tri.pts, sel, axis=0)
                pt = self._tri.pts[sel].mean(axis=0)
                pts = np.vstack([pts_unsel, pt])
                self.setData(pts)

    def _selection_changed(self, sel: np.ndarray):
        if len(sel) == 2:
            r = np.linalg.norm(self._tri.pts[sel[0]] - self._tri.pts[sel[1]])
            self._dist_lbl.setText(f'Distance: {r:.2f}')
        else:
            self._dist_lbl.setText(f'Distance: -')

    def _undo(self):
        print('undo')

    def _redo(self):
        print('redo')

class Register3DWindow(MainWindow):
    def __init__(self, path: str, *args, use_existing=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        assert os.path.isfile(path), f'{path} is not a file'
        fname, ext = os.path.splitext(path)
        assert ext in ['.txt', '.csv']
        self._path_new = f'{fname}.registered.txt'

        if os.path.isfile(self._path_new) and use_existing:
            print(f'Found previously registered points, using: {self._path_new}')
            pts = np.loadtxt(self._path_new)
        else:
            pts = np.loadtxt(path)

        assert pts.shape[1] == 3, 'Wrong number of columns'
        print(f'Loaded data of shape: {pts.shape}')

        # Widgets
        self.setWindowTitle('Register 3D')
        self._reg = Register3DWidget()
        self.setCentralWidget(self._reg)
        self._reg.setData(pts)
        self.resizeToActiveScreen()

        # Listeners
        self._reg.saved.connect(self._save)

    def _save(self):
        pts = self._reg.getData()
        assert not pts is None
        print(f'Saving {len(pts)} points to: {self._path_new}')
        np.savetxt(self._path_new, pts)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source z-stack (.tif[f])')
    args = parser.parse_args()

    # pg.setConfigOption('background', 'w')
    # pg.setConfigOption('foreground', 'k')

    app = QtWidgets.QApplication(sys.argv)
    window = Register3DWindow(args.file)
    window.show()
    sys.exit(app.exec_())