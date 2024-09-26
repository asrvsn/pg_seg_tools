'''
Manual mask-based segmentor
'''

import pickle
import skimage
import skimage.io
import os

from .widgets.seg_2d import *
from .utils.data import *

class MaskSegmentWidget(SaveableWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._plot = MaskImageWidget(editable=True)
        self._plot.setTitle('Mask Segmentation')
        self._main_layout.addWidget(self._plot)

    def setData(self, img: np.ndarray, mask):
        self._plot.setData(img, mask)
        self.setEnabled(True)

    def getData(self) -> np.ndarray:
        mask = self._plot._mask_item._mask.copy() # TODO: fix this
        return mask
    
class MaskSegmentorWindow(MainWindow):
    def __init__(self, path: str, chan: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        img = load_stack(path) # ZXYC
        img = img[:, :, :, chan]
        self._mask_path = f'{path}.mask.npy'

        # Load existing mask if exists
        if os.path.isfile(self._mask_path):
            mask = np.load(self._mask_path)
            print(f'Loaded mask from {self._mask_path}')
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint32)
        
        self._seg = MaskSegmentWidget(self)
        self.setCentralWidget(self._seg)
        self._seg.setData(img, mask)
        self.resizeToActiveScreen()
        self._seg.saved.connect(self._save)

    def _save(self):
        mask = self._seg.getData()
        np.save(self._mask_path, mask)
        print(f'Saved mask to {self._mask_path}')

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=0, help='Channel number')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = MaskSegmentorWindow(args.file, args.c)
    win.show()
    sys.exit(app.exec())