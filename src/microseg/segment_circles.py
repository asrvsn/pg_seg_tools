'''
Manual circles segmentor
'''
from .widgets.seg_2d import *
from .utils.data import *

from matgeo import Ellipsoid

class CirclesSegmentorWidget(ThingSegmentorWidget):
    def setData(self, img, things):
        # Allow casting of general ellipsoids
        if len(things) > 0 and (not type(things[0]) is LabeledCircle):
            assert isinstance(things[0], Ellipsoid)
            things = [LabeledCircle.from_ellipse(i, p) for i, p in enumerate(things)]
        super().setData(img, things)
    
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