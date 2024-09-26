'''
Manual ellipses segmentor
'''
from .widgets.seg_2d import *
from .utils.data import *

class EllipsesSegmentorWidget(ThingSegmentorWidget):
    def makeWidget(*args, **kwargs):
        return CirclesImageWidget(*args, **kwargs)

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=0, help='Channel number')
    parser.add_argument('-d', type=str, default='circles', help='Descriptor')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    seg = EllipsesSegmentorWidget()
    win = ThingsSegmentorWindow(args.file, args.c, seg, args.d)
    win.show()
    sys.exit(app.exec())