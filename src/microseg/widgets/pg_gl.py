'''
Pyqtgraph OpenGL widgets
'''
from typing import List
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSizePolicy
import pyqtgraph.opengl as gl

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

class GLHoverableScatterViewWidget(gl.GLViewWidget):
    '''
    GLView + Triangulation with hoverable 3D points using raycasting
    '''
    sel_eps: float=1e-2 # Percentage of viewport diagonal to consider a selection

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Widgets
        self._sp = gl.GLScatterPlotItem(pxMode=True)
        self._sp.setGLOptions('translucent')
        self._sp_hov = gl.GLScatterPlotItem(pxMode=True)
        self._sp_sel = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._sp)
        self.addItem(self._sp_hov)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # State
        self._pts = None
        self._proj_pts = None
        self._width, self._height = None, None
        self._diag = None
        self._reset_mouse()

    def setScatterData(self, pts: np.ndarray, *args, **kwargs):
        self._pts = pts
        self._sp.setData(pos=self._pts, *args, **kwargs)

    def _project(self):
        if not self._pts is None:
            # Get the current view and projection matrices
            proj = np.array(self.projectionMatrix(region=None).data()).reshape(4, 4)
            view = np.array(self.viewMatrix().data()).reshape(4, 4)
            mat = view @ proj

            # Project 3D points to normalized device coordinates
            pts_4d = np.column_stack((self._pts, np.ones(self._pts.shape[0])))
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
            # print('ran projection')

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        self._project()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if not self._proj_pts is None:
            pos = ev.pos()
            dists = np.linalg.norm(self._proj_pts - np.array([pos.x(), pos.y()]), axis=1)
            idx = np.argmin(dists)
            if dists[idx] < self.sel_eps * self._diag:
                hovered = idx
            else:
                hovered = None
            if hovered != self._hovered:
                self._hovered = hovered
                self._redraw_sel()

    def _redraw_sel(self):
        hov = [] if self._hovered is None else [self._hovered]
        self._sp_hov.setData(pos=self._pts[hov], size=80, color=(1,1,1,0.25))

    def _reset_mouse(self):
        self._hovered = None

class GLSelectableScatterViewWidget(GLHoverableScatterViewWidget):
    '''
    Same as hover but supports (multiple) selection of the points
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addItem(self._sp_sel)

    def _reset_mouse(self):
        super()._reset_mouse()
        self._selected = set()

    def _redraw_sel(self):
        self._sp_sel.setData(pos=self._pts[list(self._selected)], size=80, color=(1,1,1,0.5))

    def mousePressEvent(self, ev):
        super().mousePressEvent(ev)
        if not self._hovered is None:
            if ev.button() == Qt.MouseButton.LeftButton:
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._selected.add(self._hovered)
                else:
                    self._selected = set([self._hovered])
                self._redraw_sel()
        

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

