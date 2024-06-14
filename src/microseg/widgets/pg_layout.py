'''
Pyqtgraph layouts
'''

class SubplotGrid(pg.GraphicsLayoutWidget):
    '''
    Grid of 2d plots
    '''
    def __init__(self, nr: int, nc: int, synced: bool=False, **kwargs):
        super().__init__(**kwargs)
        self._layout = QtWidgets.QGridLayout()
        self._nr = nr
        self._nc = nc
        self._plots = np.array([PlotWidget() for _ in range(nr*nc)])
        if synced:
            for i in range(nr*nc):
                for j in range(i+1, nr*nc):
                    self._plots[i].setXLink(self._plots[j])
                    self._plots[i].setYLink(self._plots[j])
                    self._plots[j].setXLink(self._plots[i])
                    self._plots[j].setYLink(self._plots[i])
        self._plots = self._plots.reshape((nr, nc))
        for y in range(nr):
            for x in range(nc):
                self._layout.addWidget(self._plots[y,x], y, x)
        self.setLayout(self._layout)

    def __getitem__(self, key: slice) -> PlotWidget:
        return self._plots[key]