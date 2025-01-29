'''
Manual cell segmentor
'''
import pickle
import skimage
import skimage.io

from .widgets.seg_2d import *
from .utils.data import get_voxel_size

class CellSegmentorWidget(SaveableWidget):
    exported = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._editor = CirclesImageWidget(editable=True)
        self._main_layout.addWidget(self._editor)
        self._z_slider = IntegerSlider(mode='scroll')
        self._settings_layout.addWidget(self._z_slider)
        self._exp_btn = PushButton('Export')
        self._settings_layout.addWidget(self._exp_btn)

        # Listeners
        self._z_slider.valueChanged.connect(lambda z: self._setZ(z, slide=False))
        self._l_sc = QShortcut(QKeySequence('Left'), self)
        self._l_sc.activated.connect(lambda: self._advance(-1))
        self._r_sc = QShortcut(QKeySequence('Right'), self)
        self._r_sc.activated.connect(lambda: self._advance(1))
        self._sl_sc = QShortcut(QKeySequence('Shift+Left'), self)
        self._sl_sc.activated.connect(lambda: self._extrude(-1))
        self._sr_sc = QShortcut(QKeySequence('Shift+Right'), self)
        self._sr_sc.activated.connect(lambda: self._extrude(1))
        self._csl_sc = QShortcut(QKeySequence('Ctrl+Shift+Left'), self)
        self._csl_sc.activated.connect(lambda: self._intrude(-1))
        self._csr_sc = QShortcut(QKeySequence('Ctrl+Shift+Right'), self)
        self._csr_sc.activated.connect(lambda: self._intrude(1))
        self._exp_btn.clicked.connect(self._export)
        self._editor.edited.connect(self._edited)

        # State
        self._z = 0
        self._ZXY = None
        self._circles = None

    def setData(self, ZXY: np.ndarray, circles: np.ndarray=None):
        self._ZXY = ZXY
        if circles is None:
            circles = [[] for _ in range(ZXY.shape[0])]
        self._circles = circles
        self._setZ(0)
        self.setDisabled(False)
        self._advance_label()

    def _advance(self, dz: int):
        # Advance frame with selection maintenance
        z = max(0, min(self._ZXY.shape[0]-1, self._z + dz))
        if z != self._z:
            i = self._editor._selected
            circ = None
            if not (i is None) and i < len(self._circles[self._z]):
                circ = self._circles[self._z][i]
            self._setZ(z, slide=True)
            if not (circ is None):
                try:
                    self._editor._select(self._circles[z].index(circ))
                except ValueError:
                    pass
    
    def _setZ(self, z: int, slide: bool=True):
        self._z = z
        if slide:
            self._z_slider.setData(0, self._ZXY.shape[0]-1, z)
        self._editor.setImage(self._ZXY[z])
        self._editor.setCircles(self._circles[z])

    def _extrude(self, dz: int):
        assert dz in [-1, 1], 'dz must be -1 or 1'
        i = self._editor._selected
        if not i is None:
            left = dz == -1 and self._z > 0
            right = dz == 1 and self._z < self._ZXY.shape[0]-1
            if left or right:
                circ = self._circles[self._z][i]
                z_ = self._z + dz
                if not any(c == circ for c in self._circles[z_]):
                    print(f'No duplicate found, adding circle')
                    self._circles[z_].append(circ.copy())
                self._advance(dz)
    
    def _intrude(self, dz: int):
        assert dz in [-1, 1], 'dz must be -1 or 1'
        i = self._editor._selected
        if not i is None:
            z = self._z
            self._advance(dz)
            self._circles[z].pop(i)
            if z == self._z:
                self._setZ(z) # Redraw so we see it
            print(f'Removed circle')

    def _export(self):
        self.exported.emit()

    def _edited(self):
        self._circles[self._z] = [c.copy() for c in self._editor._circles]
        self._advance_label()
        print('circles edited')

    def _advance_label(self):
        l = 0
        for cs in self._circles:
            for c in cs:
                l = max(l, c.l)
        l += 1
        self._editor._next_label = l

    def getData(self) -> List[LabeledCircle]:
        return self._circles

class CellSegmentorWindow(MainWindow):
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        assert os.path.isfile(path), f'{path} is not a file'
        fname, ext = os.path.splitext(path)
        assert ext in ['.tif', '.tiff'], f'{path} is not a .tif file'
        self._save_path = f'{fname}.seg'
        self._export_path = f'{fname}.csv'

        ZXY = skimage.io.imread(path)
        assert ZXY.ndim == 3, f'{path} is not a 3D image'
        assert ZXY.shape[1] == ZXY.shape[2], f'{path} is not a square ZXY image'
        print(f'Opening file of shape: {ZXY.shape}')
        self._voxres = get_voxel_size(path, fmt='XYZ')
        print(f'Physical pixel sizes (x,y,z): {self._voxres}')

        circles = None
        if os.path.isfile(self._save_path):
            print(f'Loading segmentation from {self._save_path}')
            circles = np.load(self._save_path, allow_pickle=True)
            assert len(circles) == ZXY.shape[0], 'Wrong number of circles'
        
        pg.setConfigOptions(antialias=True, useOpenGL=False) 
        self.setWindowTitle('Cell Segmentor')
        self._seg = CellSegmentorWidget()
        self.setCentralWidget(self._seg)
        self._seg.setData(ZXY, circles=circles)
        self.resizeToActiveScreen()

        # Listeners
        self._seg.saved.connect(self._save)
        self._seg.exported.connect(self._export)

    def _save(self):
        circles = self._seg.getData()
        pickle.dump(circles, open(self._save_path, 'wb'))
        print(f'Saved segmentation to {self._save_path}')

    def _export(self):
        centroids = dict()
        circles = self._seg.getData()
        for z, cs in enumerate(circles):
            for c in cs:
                v = np.concatenate((c.v, [z])) # Add z value
                if c.l in centroids:
                    centroids[c.l].append(v)
                else:
                    centroids[c.l] = [v]
        if len(centroids) > 0:
            centroids = np.array([np.array(vs).mean(axis=0) for vs in centroids.values()])
            centroids *= self._voxres
            np.savetxt(self._export_path, centroids)
            print(f'Exported centroids to {self._export_path}')
        

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source z-stack (.tif[f])')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = CellSegmentorWindow(args.file)
    win.show()
    sys.exit(app.exec())