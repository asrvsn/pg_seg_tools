'''
Pyqtgraph pens, brushes, etc.
'''
import pyqtgraph as pg
from qtpy import QtCore

from .colors import *

cc_pens = np.array([
    pg.mkPen(*rgb, width=5) for rgb in cc_glasbey_255
])

cc_pens_hover = np.array([
    pg.mkPen(*rgb, width=10) for rgb in cc_glasbey_255
])

cc_pens_dashed = np.array([
    pg.mkPen(*rgb, width=5, style=QtCore.Qt.DashLine) for rgb in cc_glasbey_255
])

n_pens = len(cc_pens)