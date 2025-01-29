'''
Base classes for building apps from ROI editors
'''
import pyqtgraph.opengl as gl # Has to be imported before qtpy
from matgeo import Triangulation

from .base import *
from .pg import *
from .roi_image import *
from .pg_gl import *

class ImageSegmentorApp(SaveableApp):
    '''
    Simple usable app for segmenting single (or stacks) of images in ZXYC format
    '''
    def __init__(self, img_path: str, desc: str='rois', *args, **kwargs):
        # State
        self._z = 0
        self._img_path = img_path
        self._img = load_stack(img_path, fmt='ZXYC')
        self._zmax = self._img.shape[0]
        self._rois = [[] for _ in range(self._zmax)]
        
        # Widgets
        self._main = VLayoutWidget()
        self._creator = ROIsCreator()
        self._main._layout.addWidget(self._creator)
        self._z_slider = IntegerSlider(mode='scroll')
        self._main._layout.addWidget(self._z_slider)
        if self._zmax == 1:
            print(f'Received standard 2D image, disabling z-slider')
            self._z_slider.hide()
        else:
            print(f'Received z-stack with {self._zmax} slices, enabling z-slider')
            self._z_slider.setData(0, self._zmax-1, self._z)
        self._creator.setImage(self._img[self._z])

        # Listeners
        self._creator.add.connect(self._add)
        self._creator.delete.connect(self._delete)
        self._z_slider.valueChanged.connect(lambda z: self._set_z(z))

        # Run data load and rest of initialization in superclass
        super().__init__(
            f'Segmenting {desc} on image: {os.path.basename(img_path)}',
            f'{os.path.splitext(img_path)[0]}.{desc}',
        *args, **kwargs)
        self.setCentralWidget(self._main)

    ''' Overrides '''

    def copyIntoState(self, state: List[List[ROI]]):
        self._rois = [[r.copy() for r in subrois] for subrois in state]
        assert len(self._rois) == self._zmax, f'Expected {self._zmax} z-slices, got {len(self._rois)}'
        self._refresh_ROIs()

    def copyFromState(self) -> List[List[ROI]]:
        return [[r.copy() for r in subrois] for subrois in self._rois]

    def readData(self, path: str) -> List[List[ROI]]:
        rois = pickle.load(open(path, 'rb'))
        # Allow flat-list of ROIs for 1-stack images
        if not type(rois[0]) is list:
            rois = [rois]
        # Allow unlabeled ROIs
        lbl = max([max([r.lbl for r in subrois if type(r) is LabeledROI], default=0) for subrois in rois], default=0) + 1
        for subrois in rois:
            for i, r in enumerate(subrois):
                if not type(r) is LabeledROI:
                    subrois[i] = LabeledROI(lbl, r)
                    lbl += 1
        return rois
    
    def writeData(self, path: str, rois: List[List[ROI]]):
        # Write flat-list of ROIs for 1-stack images
        if self._zmax == 1:
            rois = rois[0]
        pickle.dump(rois, open(path, 'wb'))

    def keyPressEvent(self, evt):
        if evt.key() == Qt.Key_Left:
            self._set_z(self._z-1, set_slider=True)
        elif evt.key() == Qt.Key_Right:
            self._set_z(self._z+1, set_slider=True)
        else:
            super().keyPressEvent(evt)

    ''' Private methods '''

    def _refresh_ROIs(self):
        self._creator.setROIs(self._rois[self._z])

    def _set_z(self, z: int, set_slider: bool=False):
        z = max(0, min(z, self._zmax-1))
        if z != self._z:
            if set_slider:
                self._z_slider.setValue(z) # Callback will go to next branch
            else:
                self._set_z_raw(z)

    def _set_z_raw(self, z: int):
        # Hook-in for children
        self._z = z
        self._creator.setData(self._img[z], self._rois[z])

    @property
    def next_label(self) -> int:
        return max(
            [max([r.lbl for r in subrois], default=-1) 
             for subrois in self._rois], 
            default=-1
        ) + 1

    def _add(self, rois: List[ROI]):
        l = self.next_label
        lrois = [LabeledROI(l+i, r) for i, r in enumerate(rois)]
        self._rois[self._z].extend(lrois)
        self._refresh_ROIs()
        self.pushEdit()

    def _delete(self, rois: Set[int]):
        self._rois[self._z] = [
            r for r in self._rois[self._z] if not (r.lbl in rois)
        ]
        self._refresh_ROIs()
        self.pushEdit()

class ZStackObjectViewer(gl.GLViewWidget):
    '''
    3D viewer of multiple objects with current z-plane rendered
    '''
    mesh_opts: dict = {
        'shader': 'normalColor',
        'glOptions': 'opaque',
        'drawEdges': True,
        'smooth': False,
    }
    grid_opts: dict = {
        # 'glOptions': 'translucent',
        # 'color': (0.5, 0.5, 0.5, 1.0),
    }
    cursor_opts: dict = {
        'pxMode': True,
        'color': (1.0, 1.0, 1.0, 1.0),
        'size': 10,
    }
    facecolors = cc_glasbey_01_rgba

    def __init__(self, shape_x: float, shape_y: float, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Z-Slice Object Viewer')
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        MainWindow.resizeToScreen(self, offset=1) # Show on next avail screen
        self.setCameraPosition(distance=max(shape_x, shape_y)*1.5) # Zoom out sufficiently
        self._plane = GLPlaneItem(shape_x, shape_y)
        self.addItem(self._plane)
        self._cursor_pt = gl.GLScatterPlotItem(**self.cursor_opts)
        self.addItem(self._cursor_pt)
        self._meshes = []

    def setObjects(self, objs: List[Triangulation], labels: List[int]):
        for mesh in self._meshes:
            self.removeItem(mesh)
        self._meshes = []
        for obj, lbl in zip(objs, labels):
            colors = np.full((len(obj.simplices), 4), self.facecolors[lbl])
            md = gl.MeshData(vertexes=obj.pts, faces=obj.simplices, faceColors=colors)
            mesh = gl.GLMeshItem(meshdata=md, **self.mesh_opts)
            self.addItem(mesh)
            self._meshes.append(mesh)

    def setZ(self, z: float):
        self._plane.setZ(z)

    def setXY(self, xy: Tuple[int, int]):
        self._cursor_xy = xy
        self._cursor_pt.setData(pos=[xy])

    def closeEvent(self, evt):
        for widget in QApplication.instance().topLevelWidgets(): # Intercept the close evt and close the main application.
            if isinstance(widget, ZStackSegmentorApp):  
                widget.close()
                break  # Close only the first detected instance
        evt.accept()  

class ZStackSegmentorApp(ImageSegmentorApp):
    '''
    Segment 3-dimensional structures using iterated 2-dimensional segmentation and fusion/registration
    TODO:
    1. Add an auxiliary window which shows the current 3D structure alone
    2. Add a registration step between successive slices which allows:
        a. Manual association of ROIs to the same label
        b. Automated associations of ROIs
        c. Both in a kind of "proposer" mode as in the 2D case
        d. Proposer mode disables edits at the 2D level, but bubbles up selections to 3D for association
        e. Re-label ROIs when associated to min label
    3. Upon registration, re-compute and show the 3D structure
        a. Recompute the 3D structure using a variety of options, e.g. convex hull, advancing front, etc from Triangulation lib
            - make sure to use correct z-position using image metadata (voxel aspect ratio)
        b. Render current z-plane in view
        c. Render position of cursor in z-plane (might need to bubble up from lower level)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._voxsize = get_voxel_size(self._img_path, fmt='XYZ') # Physical voxel sizes
        shape_x, shape_y = self._voxsize[:2] * np.array(self._img.shape[1:3])
        self._viewer = ZStackObjectViewer(shape_x, shape_y)
        self._viewer.show()

    def _set_z_raw(self, z: int):
        super()._set_z_raw(z)
        self._viewer.setZ(z * self._voxsize[2])

    def closeEvent(self, evt):
        if not self._viewer is None:
            self._viewer.close() 
        evt.accept()