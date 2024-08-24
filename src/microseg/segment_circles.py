'''
Manual cell segmentor
'''
import pickle
import os

from .widgets.seg_2d import *
from .utils.data import *

class CirclesSegmentorWidget(SaveableWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._editor = CirclesImageWidget(editable=True, limit_zoom=False)
        self._main_layout.addWidget(self._editor)

        # Listeners
        self._editor.edited.connect(self._edited)

        # State
        self._img: np.ndarray = None
        self._circles: List[LabeledCircle] = []

    def setData(self, img: np.ndarray, circles: List[LabeledCircle]=[]):
        self._img = img
        self._circles = circles
        self._editor.setImage(img)
        self._editor.setCircles(circles)
        self._advance_label()
        self.setDisabled(False)

    def _advance_label(self):
        l = 0
        for c in self._circles:
            l = max(l, c.l)
        l += 1
        self._editor._next_label = l

    def _edited(self):
        self._circles = self._editor.getThings()
        self._advance_label()
        print('circles edited')

    def getData(self):
        return self._circles

class CirclesSegmentorWindow(MainWindow):

    def __init__(self, path: str, chan: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        img = load_stack(path)[chan]
        self._circ_path = f'{path}.circles'

        # Load existing circles if exists
        circles = []
        if os.path.isfile(self._circ_path):
            print(f'Loading circles from {self._circ_path}')
            circles = pickle.load(open(self._circ_path, 'rb'))

        pg.setConfigOptions(antialias=True, useOpenGL=False) 
        self.setWindowTitle('Circles Segmentor')
        self._seg = CirclesSegmentorWidget()
        self.setCentralWidget(self._seg)
        self._seg.setData(img, circles=circles)
        self.resizeToActiveScreen()

        # Listeners
        self._seg.saved.connect(self._save)

    def _save(self):
        circles = self._seg.getData()
        pickle.dump(circles, open(self._circ_path, 'wb'))
        print(f'Saved circles to {self._circ_path}')

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=0, help='Channel number')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = CirclesSegmentorWindow(args.file, args.c)
    win.show()
    sys.exit(app.exec())