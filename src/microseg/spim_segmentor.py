'''
2D segmentor / curator for sets of SPIM z-stacks in the form of .czi files
- assumes CZI format
- uses cellpose for auto-segmentation
- custom gui for manual curation
'''

import pyqtgraph as pg
from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QShortcut, QLabel, QTableWidgetItem, QGridLayout
from qtpy.QtGui import QKeySequence

import matplotlib.pyplot as plt
from aicsimageio import AICSImage
import pickle
import time

from seg_2d import *
from pg_seg_widgets import *
from plane import *
from mask_utils import *

class SegmentationInfoWidget(QtWidgets.QWidget):
    '''
    Display info about the current segmentation
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        # self._layout.addWidget(QLabel('Segmentation info:'))
        ## Overall info
        self._overall = StretchTableWidget(self)
        self._overall.setColumnCount(2)
        nrows = 4
        self._overall.setRowCount(nrows)
        self._overall.setItem(0, 0, QTableWidgetItem('Shape (CZXY):'))
        self._overall.setItem(1, 0, QTableWidgetItem('Units per pixel (XY):'))
        self._overall.setItem(2, 0, QTableWidgetItem('Last modified:'))
        self._overall.setItem(3, 0, QTableWidgetItem('Circular radius:'))
        for i in range(nrows):
            self._overall.item(i, 0).setTextAlignment(Qt.AlignLeft)
            self._overall.setItem(i, 1, QTableWidgetItem(''))
            self._overall.item(i, 1).setTextAlignment(Qt.AlignRight)
        self._layout.addWidget(self._overall)
        ## Per-channel info
        self._per_channel = StretchTableWidget(self) # Dummy until segmentation is set
        self._layout.addWidget(self._per_channel)

        # Listeners

        # State
        self._seg = None
        self._seg_path = None
        self._last_modified = 'Never'

    def setSeg(self, seg: Segmentation2D, seg_path: str):
        self._seg = seg
        self._seg_path = seg_path
        self._last_modified = time.ctime(os.path.getmtime(self._seg_path)) if os.path.isfile(self._seg_path) else 'Never'
        self._draw_seg()

    def _draw_seg(self):
        # Update overall info
        self._overall.item(0, 1).setText(str(self._seg.czxy))
        self._overall.item(1, 1).setText(str(self._seg.upp))
        self._overall.item(2, 1).setText(self._last_modified)
        self._overall.item(3, 1).setText(str(self._seg.circular_radius))
        # Remove and replace per-channel info table based on # channels
        self._layout.removeWidget(self._per_channel)
        self._per_channel.deleteLater()
        self._per_channel = StretchTableWidget(self)
        nc = self._seg.czxy[0]
        self._per_channel.setColumnCount(nc + 1)
        nrows = 11
        self._per_channel.setRowCount(nrows)
        self._per_channel.setItem(0, 0, QTableWidgetItem('Channel:'))
        self._per_channel.setItem(1, 0, QTableWidgetItem('Z projection mode:'))
        self._per_channel.setItem(2, 0, QTableWidgetItem('Z projection window:'))
        self._per_channel.setItem(3, 0, QTableWidgetItem('Denoise mode:'))
        self._per_channel.setItem(4, 0, QTableWidgetItem('Cellpose model:'))
        self._per_channel.setItem(5, 0, QTableWidgetItem('Nuclear channel:'))
        self._per_channel.setItem(6, 0, QTableWidgetItem('Diameter:'))
        self._per_channel.setItem(7, 0, QTableWidgetItem('Flow threshold:'))
        self._per_channel.setItem(8, 0, QTableWidgetItem('Cellprob threshold:'))
        self._per_channel.setItem(9, 0, QTableWidgetItem('Cellpose objects:'))
        self._per_channel.setItem(10, 0, QTableWidgetItem('Curated objects:'))
        for i in range(nrows):
            self._per_channel.item(i, 0).setTextAlignment(Qt.AlignLeft)
        for j in range(nc):
            self._per_channel.setItem(0, j+1, QTableWidgetItem(str(j)))
            self._per_channel.setItem(1, j+1, QTableWidgetItem(self._seg.settings.zproj_mode[j]))
            self._per_channel.setItem(2, j+1, QTableWidgetItem(str(self._seg.settings.zproj_window[j])))
            self._per_channel.setItem(3, j+1, QTableWidgetItem(self._seg.settings.cp_denoise[j]))
            self._per_channel.setItem(4, j+1, QTableWidgetItem(self._seg.settings.cp_model[j]))
            self._per_channel.setItem(5, j+1, QTableWidgetItem(str(self._seg.settings.cp_nuclear_channel[j])))
            self._per_channel.setItem(6, j+1, QTableWidgetItem(str(self._seg.settings.cp_diam[j])))
            self._per_channel.setItem(7, j+1, QTableWidgetItem(str(self._seg.settings.cp_flow[j])))
            self._per_channel.setItem(8, j+1, QTableWidgetItem(str(self._seg.settings.cp_cellprob[j])))
            nlabels_cp = len(np.unique(self._seg.cp_mask[j])) - 1
            nlabels_curated = len(np.unique(self._seg.mask[j])) - 1
            self._per_channel.setItem(9, j+1, QTableWidgetItem(str(nlabels_cp)))
            self._per_channel.setItem(10, j+1, QTableWidgetItem(str(nlabels_curated)))
            for i in range(nrows):
                self._per_channel.item(i, j+1).setTextAlignment(Qt.AlignRight)
        self._layout.addWidget(self._per_channel)

class SegmentationWidget(QtWidgets.QWidget):
    channels = ['r', 'g', 'b']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Widgets
        self._layout = QGridLayout()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)
        self._zproj = ZProjectWidget()
        self._mask_cp = CellposeWidget()
        self._mask_curated = CuratorWidget()
        self._info = SegmentationInfoWidget()
        self._layout.addWidget(self._zproj, 0, 0)
        self._layout.addWidget(self._mask_cp, 0, 1)
        self._layout.addWidget(self._info, 1, 0)
        self._layout.addWidget(self._mask_curated, 1, 1)

        # Listeners
        link_plots(self._zproj._plot, self._mask_cp._plot)
        link_plots(self._zproj._plot, self._mask_curated._plot)
        link_plots(self._mask_cp._plot, self._mask_curated._plot)
        self._shortcuts = []
        for c in self.channels:
            self._add_channel_shortcut(c)
        self._zproj.saved.connect(lambda: self._save_zproj())
        self._mask_cp.saved.connect(lambda: self._save_cp())
        self._mask_curated.saved.connect(lambda: self._save_curated())
        # self._zproj.draw_outline.connect(self._draw_outline)

        # State
        self._data = None
        self._c = 0
        self._seg = None
        self._seg_path = None

    def _add_channel_shortcut(self, c: str):
        self._add_shortcut(c, lambda: self._change_channel(self.channels.index(c)))

    def _add_shortcut(self, key: str, fun: Callable):
        self._shortcuts.append(QShortcut(QKeySequence(key), self))
        self._shortcuts[-1].activated.connect(fun)

    def _change_channel(self, ch: int):
        self._c = ch
        self._zproj.setData(
            self._data[self._c], 
            self._seg.settings.zproj_mode[self._c],
            self._seg.settings.zproj_window[self._c],
        )
        self._mask_cp.setData(
            self._seg.zproj,
            self._seg.img,
            self._c,
            self._seg.cp_mask[self._c],
            self._seg.settings.cp_denoise[self._c],
            self._seg.settings.cp_model[self._c],
            self._seg.settings.cp_nuclear_channel[self._c],
            self._seg.settings.cp_diam[self._c],
            self._seg.settings.cp_flow[self._c],
            self._seg.settings.cp_cellprob[self._c],
        )
        mask_outline = np.zeros_like(self._seg.mask[self._c])
        if self._seg.outline is not None:
            draw_poly(mask_outline, self._seg.outline.vertices.flatten().tolist(), 1)
        self._mask_curated.setData(
            self._seg.zproj[self._c], # View raw data for manual curation
            self._seg.cp_mask[self._c],
            self._seg.mask[self._c],
            mask_outline,
        )

    def _draw_outline(self, outline: Optional[PlanarPolygon]):
        self._seg.outline = outline

    def _save_zproj(self):
        proj, mode, window = self._zproj.getData()
        self._seg.zproj[self._c] = proj.astype(self._seg.zproj.dtype)
        self._seg.img[self._c] = proj.astype(self._seg.img.dtype) # Update underlying img as well
        self._seg.settings.cp_denoise[self._c] = AVAIL_DENOISE_MODES[0] # Reset denoise mode when underlying img is updated
        self._seg.settings.zproj_mode[self._c] = mode
        self._seg.settings.zproj_window[self._c] = window
        self._save_seg()
        print(f'Saved z projection with mode {mode} and window {window}')
        self._refresh()

    def _save_cp(self):
        img, mask, denoise, model, nc, diam, flow, cellprob = self._mask_cp.getData()
        self._seg.img[self._c] = img.astype(self._seg.img.dtype)
        self._seg.cp_mask[self._c] = mask.astype(self._seg.cp_mask.dtype)
        self._seg.settings.cp_denoise[self._c] = denoise
        self._seg.settings.cp_model[self._c] = model
        self._seg.settings.cp_nuclear_channel[self._c] = nc
        self._seg.settings.cp_diam[self._c] = diam
        self._seg.settings.cp_flow[self._c] = flow
        self._seg.settings.cp_cellprob[self._c] = cellprob
        self._save_seg()
        print(f'Saved cellpose mask with denoise {denoise}, model {model}, nuclear channel {nc}, and diameter {diam}')
        self._refresh()

    def _save_curated(self):
        mask, mask_outline = self._mask_curated.getData()
        outline_polys = [PlanarPolygon(p, use_chull_if_invalid=True) for p in mask_to_polygons(mask_outline)]
        assert len(outline_polys) <= 1, 'Cannot have more than one outline'
        print(f'Saving {len(outline_polys)} outlines')
        self._seg.mask[self._c] = mask.astype(self._seg.mask.dtype)
        self._seg.outline = outline_polys[0] if len(outline_polys) == 1 else None
        self._save_seg()
        print(f'Saved curated mask')
        self._refresh()

    def _save_seg(self):
        pickle.dump(self._seg, open(self._seg_path, 'wb'))
        print(f'Saved segmentation to {self._seg_path}')
        self._refresh_info()

    def _refresh(self):
        self.setData(self._data, self._seg, self._seg_path, ch=self._c)

    def _refresh_info(self):
        self._info.setSeg(self._seg, self._seg_path)

    def setData(self, data: np.ndarray, seg: Segmentation2D, seg_path: str, ch: int=0):
        assert data.ndim == 4, 'data must be CZXY'
        assert data.shape[0] == len(self.channels), 'data must have same number of channels as SegmentationWidget.channels'
        self._data = data
        self._seg = seg
        self._seg_path = seg_path
        self._refresh_info()
        self._change_channel(ch)

class SegmentorWidget(QtWidgets.QSplitter):
    sizes = [200, 800]
    extension = 'czi'
    seg_extension = 'seg'
    
    def __init__(self, folder: str, *args, **kwargs):
        super().__init__(Qt.Horizontal, *args, **kwargs)

        # Widgets
        self._tree = ExtensionViewer(folder, [f'*.{self.extension}'])
        self._seg = SegmentationWidget()
        self.addWidget(self._tree)
        self.addWidget(self._seg)
        self.setSizes(self.sizes)
        self._tree.file_selected.connect(self.setData)

    def setData(self, path: str):
        assert os.path.isfile(path), f'{path} is not a file'
        assert os.path.splitext(path)[1] == f'.{self.extension}', f'{path} is not a .{self.extension} file'
        img = AICSImage(path)
        data = img.get_image_data('CZYX') # Y/X axes are transposed
        print(f'Loading {path} with shape {data.shape}')
        assert len(img.physical_pixel_sizes) == 3, 'image must be z-stack'
        seg_path = os.path.splitext(path)[0] + f'.{self.seg_extension}'
        if os.path.isfile(seg_path):
            seg = pickle.load(open(seg_path, 'rb'))
            assert isinstance(seg, Segmentation2D), f'Expected {seg_path} to contain a Segmentation2D object'
        else:
            upp = img.physical_pixel_sizes[1:]
            seg = Segmentation2D(data.shape, upp)
        self._seg.setData(data, seg, seg_path)

class SegmentorWindow(MainWindow):
    def __init__(self, folder: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pg.setConfigOptions(antialias=True, useOpenGL=False) 
        self.setWindowTitle('SPIM Segmentor')
        self.setCentralWidget(SegmentorWidget(folder))
        self.resizeToActiveScreen()

if __name__ == '__main__':
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='folder containing .czi files')
    args = parser.parse_args()

    assert os.path.isdir(args.folder), 'folder must be a directory'

    app = QtWidgets.QApplication(sys.argv)
    win = SegmentorWindow(args.folder)
    win.show()
    sys.exit(app.exec())