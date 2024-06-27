import matplotlib.pyplot as plt
import cv2
import pyqtgraph as pg

from pg_seg_widgets import *

class VideoSegmentorWidget(ZStackWidget):
    pass

class VideoSegmentorWindow(MainWindow):
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        assert os.path.isfile(path), f'{path} is not a file'
        fname, ext = os.path.splitext(path)
        assert ext in ['.avi'], f'{path} is not a .avi file'
        self._cap = cv2.VideoCapture(path) 

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f'Frames per second: {fps}')
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frame count: {frame_count}')
        duration = frame_count / fps
        print(f'Duration: {duration} seconds')

        # Grab nth-mth frames
        ZXY = []
        m, n = 0, 10
        for i in range(m, n):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self._cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ZXY.append(frame)
        ZXY = np.stack(ZXY, axis=0)
        print(f'Shape: {ZXY.shape}')

        pg.setConfigOptions(antialias=True, useOpenGL=False) 
        self.setWindowTitle('Cell Segmentor')
        self._seg = VideoSegmentorWidget()
        self.setCentralWidget(self._seg)
        self._seg.setData(ZXY, 0)
        self.resizeToActiveScreen()

if __name__ == '__main__':
    path = '/Users/asrvsn/Nextcloud/phd/data/rotifer/MV-CH650-90TM-F-NF (J91482126)_comp.avi'
    app = QtWidgets.QApplication(sys.argv)
    win = VideoSegmentorWindow(path)
    win.show()
    sys.exit(app.exec())