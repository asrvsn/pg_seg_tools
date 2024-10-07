'''
Manual polygon-based segmentor
'''

from .widgets.seg_2d import *
from .utils.data import *

from matgeo.plane import PlanarPolygon

class PolysSegmentorWidget(ThingSegmentorWidget):
    def setData(self, img, things):
        # Allow polygons or labeled polygons
        if len(things) > 0 and type(things[0]) is PlanarPolygon:
            things = [LabeledPolygon.from_poly(i, p) for i, p in enumerate(things)]
        super().setData(img, things)
    
    def makeWidget(*args, **kwargs):
        return PolysImageWidget(*args, **kwargs)
    
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img')
    parser.add_argument('-c', type=int, default=None, help='Channel number')
    parser.add_argument('-d', type=str, default='polygons', help='Descriptor')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    seg = PolysSegmentorWidget()
    win = ThingsSegmentorWindow(args.file, seg, chan=args.c, desc=args.d)
    win.show()
    sys.exit(app.exec())