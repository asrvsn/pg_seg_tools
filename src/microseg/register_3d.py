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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._vw = GLSelectableSurfaceViewWidget()
        self._main_layout.addWidget(self._vw)

        # Listeners

        # State
        self._tri = None
        self._lbls = None

    def setData(self, pts: np.ndarray, lbls: np.ndarray):
        self.setDisabled(False)
        pts -= pts.mean(axis=0) # Re-center
        self._lbls = lbls
        self._tri = Triangulation.surface_3d(pts, method='advancing_front')
        self._redraw()

    def getData(self) -> np.ndarray:
        return self._lbls.copy()

    def _redraw(self):
        if not (self._tri is None or self._lbls is None):
            colors = map_colors(self._lbls, 'categorical', rgba=True)
            self._vw.setMeshData(self._tri, color=colors)

class Register3DWindow(MainWindow):
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        assert os.path.isfile(path), f'{path} is not a file'
        fname, ext = os.path.splitext(path)
        assert ext in ['.txt', '.csv']
        self._path_lbls = f'{fname}.lbls.txt'

        pts = np.loadtxt(path)
        assert pts.shape[1] == 3, 'Wrong number of columns'
        print(f'Loaded data of shape: {pts.shape}')

        if os.path.isfile(self._path_lbls):
            lbls = np.loadtxt(self._path_lbls).astype(int)
            assert len(lbls) == len(pts), 'Wrong number of labels'
            assert len(lbls.shape) == 1
        else:
            lbls = np.arange(len(pts), dtype=int)

        # Widgets
        self.setWindowTitle('Register 3D')
        self._reg = Register3DWidget()
        self.setCentralWidget(self._reg)
        self._reg.setData(pts, lbls)
        self.resizeToActiveScreen()

        # Listeners
        self._reg.saved.connect(self._save)

    def _save(self):
        lbls = self._reg.getData()
        assert not lbls is None
        N = len(np.unique(lbls))
        print(f'Saving {N} distinct labels')
        np.savetxt(self._path_lbls, lbls.astype(int), fmt='%i')

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