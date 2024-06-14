'''
Pyqtgraph pens, brushes, etc.
'''
import pyqtgraph as pg

from colorwheel import *

cc_pens = np.array([
    pg.mkPen(*rgb, width=3) for rgb in cc_glasbey_255
])

cc_pens_hover = np.array([
    pg.mkPen(*rgb, width=6) for rgb in cc_glasbey_255
])