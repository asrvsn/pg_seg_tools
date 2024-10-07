'''
Manual circles segmentor
'''
from .widgets.seg_2d import *
from .utils.data import *

class CirclesSegmentorWidget(ThingSegmentorWidget):
    def makeWidget(*args, **kwargs):
        return CirclesImageWidget(*args, **kwargs)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=None, help='Channel number')
    parser.add_argument('-d', type=str, default='circles', help='Descriptor')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    seg = CirclesSegmentorWidget()
    win = ThingsSegmentorWindow(args.file, seg, chan=args.c, desc=args.d)
    win.show()
    sys.exit(app.exec())