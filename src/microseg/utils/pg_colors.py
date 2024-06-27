'''
Pyqtgraph pens, brushes, etc.
'''
import pyqtgraph as pg

from .colors import *

cc_pens = np.array([
    pg.mkPen(*rgb, width=3) for rgb in cc_glasbey_255
])

cc_pens_hover = np.array([
    pg.mkPen(*rgb, width=6) for rgb in cc_glasbey_255
])

n_pens = len(cc_pens)