'''
Image segmentor
'''
if __name__ == '__main__':
    import sys
    import argparse
    
    from .widgets.roi_apps import *

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source img [tiff|jpg|png|czi|...]')
    parser.add_argument('-d', type=str, default='polygons', help='Descriptor')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = ImageSegmentorApp(args.file, desc=args.d)
    win.show()
    sys.exit(app.exec())