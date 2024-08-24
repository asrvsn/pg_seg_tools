'''
Manual polygon-based segmentor
'''

from .widgets.seg_2d import *
from .utils.data import *

class PolysSegmentorWidget(ThingSegmentorWidget):
    def makeWidget(*args, **kwargs):
        return PolysImageWidget(*args, **kwargs)
    
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=0, help='Channel number')
    parser.add_argument('-d', type=str, default='polygons', help='Descriptor')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    seg = PolysSegmentorWidget()
    win = ThingsSegmentorWindow(args.file, args.c, seg, args.d)
    win.show()
    sys.exit(app.exec())