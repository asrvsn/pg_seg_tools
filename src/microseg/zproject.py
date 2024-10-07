'''
Widget for computing a particular z-projection (save as tiff)
'''
from .utils.data import *

class ZProjectorWindow(MainWindow):
    def __init__(self, path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        self._main_layout.addWidget(ZProjectorWidget(self._path))

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to source z-stack')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = ZProjectorWindow(args.file)
    win.show()
    sys.exit(app.exec())