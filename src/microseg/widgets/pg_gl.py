'''
Pyqtgraph OpenGL widgets
'''
from typing import List
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSizePolicy, QShortcut
from qtpy.QtGui import QKeySequence
import pyqtgraph.opengl as gl

from matgeo import Triangulation

class GrabbableGLViewWidget(gl.GLViewWidget):
    '''
    Screen grabs on press enter key
    '''
    def _grab(self):
        self.grabFramebuffer().save('screenshot.png')
        print('Screenshot saved as screenshot.png')

    def keyPressEvent(self, ev):
        # Grab on enter key

        if ev.key() == Qt.Key.Key_Return:
            self._grab()
        else:
            super().keyPressEvent(ev)

class GLHoverableSurfaceViewWidget(gl.GLViewWidget):
    '''
    GLView + Triangulation with hoverable 3D points using raycasting
    '''
    sel_eps: float=1e-2 # Percentage of viewport diagonal to consider a selection
    size: int=20
    h_size: int=20
    h_alpha: float=0.75
    face_rgba: float=0.8
    mesh_opts: dict = {
        'shader': 'normalColor',
        # 'glOptions': 'opaque',
        'drawEdges': True,
        'smooth': False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._mesh = gl.GLMeshItem(**self.mesh_opts)
        self._sp_hov = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._mesh)
        self.addItem(self._sp_hov)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Listeners
        self._esc_sc = QShortcut(QKeySequence('Escape'), self)
        self._esc_sc.activated.connect(self._escape)

        # State
        self.cam_motion_enabled = True
        self._mat = None
        self._dragging_cam = False
        self._tri = None
        self._md = None
        self._proj_pts = None
        self._width, self._height = None, None
        self._diag = None
        self._cam_facing = None # Camera-facing points
        self._reset_mouse()

    def setMeshData(self, tri: Triangulation, *args, **kwargs):
        self._tri = tri
        self._md = gl.MeshData(vertexes=tri.pts, faces=tri.simplices, faceColors=np.full((len(tri.simplices), 4), self.face_rgba))
        kwargs = dict(size=self.size) | kwargs
        self._mesh.setMeshData(meshdata=self._md)

    def _project(self):
        if not self._tri is None:
            pts = self._tri.pts
            
            # Get the current view and projection matrices
            proj = np.array(self.projectionMatrix(region=None).data()).reshape(4, 4)
            view = np.array(self.viewMatrix().data()).reshape(4, 4)
            mat = view @ proj

            # If the view has change, re-project
            if self._mat is None or not np.allclose(self._mat, mat):
                self._mat = mat

                # Project 3D points to normalized device coordinates
                pts_4d = np.column_stack((pts, np.ones(pts.shape[0])))
                proj_pts = pts_4d @ mat # mat is not transposed, which is weird, but correct.

                # Perform perspective division
                proj_pts[:, :3] /= proj_pts[:, 3, None]

                # Convert to viewport coordinates
                _, __, self._width, self._height = self.getViewport()
                self._diag = np.sqrt(self._width**2 + self._height**2)
                self._proj_pts = np.column_stack((
                    (proj_pts[:, 0] + 1) * self._width / 4, # This 4 is weird, but correct.
                    (1 - proj_pts[:, 1]) * self._height / 4
                ))

                # Compute camera-facing triangles
                cam = np.array(self.cameraPosition())
                normals = self._tri.compute_normals()
                centroids = self._tri.compute_centroids()
                cam_to_tri = cam - centroids
                cam_facing_tri = (cam_to_tri * normals).sum(axis=1) < 0
                self._cam_facing = np.unique(self._tri.simplices[cam_facing_tri])

                # print('ran projection')

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        self._project()

    def mouseMoveEvent(self, ev):
        if self.cam_motion_enabled:
            super().mouseMoveEvent(ev)
            if ev.buttons():  # Check if any mouse button is held
                self._dragging_cam = True
            else:
                self._dragging_cam = False
        if not self._proj_pts is None:
            pos = ev.pos()
            dists = np.linalg.norm(self._proj_pts[self._cam_facing] - np.array([pos.x(), pos.y()]), axis=1)
            idx = np.argmin(dists)
            if dists[idx] < self.sel_eps * self._diag:
                hovered = self._cam_facing[idx]
            else:
                hovered = None
            if hovered != self._hovered:
                self._hovered = hovered
                self._redraw_sel()

    def _redraw_sel(self):
        hov = [] if self._hovered is None else [self._hovered]
        self._sp_hov.setData(pos=self._tri.pts[hov], size=self.h_size, color=(1,1,1,self.h_alpha))

    def _reset_mouse(self):
        self._hovered = None

    def _escape(self):
        self._reset_mouse()
        self._redraw_sel()

class GLSelectableSurfaceViewWidget(GLHoverableSurfaceViewWidget):
    '''
    Same as hover but supports (multiple) selection of the points
    '''
    s_alpha: float=1.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sp_sel = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._sp_sel)

    def _reset_mouse(self):
        super()._reset_mouse()
        self._selected = set()

    def _redraw_sel(self):
        super()._redraw_sel()
        self._sp_sel.setData(pos=self._tri.pts[list(self._selected)], size=self.h_size, color=(1,1,1,self.s_alpha))

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if not self._hovered is None:
            if ev.button() == Qt.MouseButton.LeftButton and not self._dragging_cam:
                print('selecting', self._hovered)
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._selected.add(self._hovered)
                else:
                    self._selected = set([self._hovered])
                self._redraw_sel()

class GLDrawableSurfaceViewWidget(GLHoverableSurfaceViewWidget):
    '''
    Same as hover but supports drawing embedded polygons
    '''
    mesh_opts: dict = GLHoverableSurfaceViewWidget.mesh_opts | {
        'shader': None
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Listeners
        self._edit_sc = QShortcut(QKeySequence('E'), self)
        self._edit_sc.activated.connect(self._edit)

        # State
        self._mask = None # For each triangle, associate a face ID (bit like a pixel mask); 0 is traditionally the unlabeled.
        self._reset_drawing_state()

    def _reset_drawing_state(self):
        self._is_drawing = False
        self._drawn_poly = None
        self._drawn_label = None
        # Enable viewport movement
        self.cam_motion_enabled = True

    def _reset_mouse(self):
        super()._reset_mouse()
        self._reset_drawing_state()

    def _edit(self):
        if not self._is_drawing:
            self._is_drawing = True
            self.cam_motion_enabled = False

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self._is_drawing:
            if not self._hovered is None:
                if Qt.MouseButton.LeftButton & ev.buttons():
                    if self._drawn_poly is None:
                        print('starting draw')
                        self._drawn_poly = [self._hovered]
                    else:
                        self._drawn_poly.append(self._hovered)

def GLMakeSynced(base_class):
    class SyncedGLViewWidget(base_class):
        '''
        Shamelessly taken from 
        https://stackoverflow.com/questions/70551355/link-cameras-positions-of-two-3d-subplots-in-pyqtgraph
        '''

        def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
            self.linked_views: List[GrabbableGLViewWidget] = []
            super().__init__(parent, devicePixelRatio, rotationMethod)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        def wheelEvent(self, ev):
            """Update view on zoom event"""
            super().wheelEvent(ev)
            self._update_views()

        def mouseMoveEvent(self, ev):
            """Update view on move event"""
            super().mouseMoveEvent(ev)
            self._update_views()

        def mouseReleaseEvent(self, ev):
            """Update view on move event"""
            super().mouseReleaseEvent(ev)
            self._update_views()

        def _update_views(self):
            """Take camera parameters and sync with all views"""
            camera_params = self.cameraParams()
            # Remove rotation, we can't update all params at once (Azimuth and Elevation)
            camera_params["rotation"] = None
            for view in self.linked_views:
                view.setCameraParams(**camera_params)

        def sync_camera_with(self, view: GrabbableGLViewWidget):
            """Add view to sync camera with"""
            self.linked_views.append(view)

    return SyncedGLViewWidget

