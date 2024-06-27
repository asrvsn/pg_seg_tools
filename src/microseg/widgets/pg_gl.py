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

class SelectableGLViewWidget(gl.GLViewWidget):
    '''
    GLView with ray casting
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ray = gl.GLLinePlotItem()
        self.addItem(self._ray)
    
    def mouseMoveEvent(self, ev):
        pos = ev.pos()
        ray = self.get_ray(pos.x(), pos.y())
        
        intersection = self.calculate_intersection(ray)
        
        if intersection is not None:
            print(f"Mouse position in data space: {intersection}")
            self.draw_ray(intersection)
        
        super().mouseMoveEvent(ev)

    def get_ray(self, x, y):
        
        _, __, width, height = self.getViewport()
        ndc_x = (2.0 * x / width) - 1.0
        ndc_y = 1.0 - (2.0 * y / height)
        
        ray_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        
        proj_matrix = np.array(self.projectionMatrix(region=None).data()).reshape(4, 4)
        ray_eye = np.linalg.inv(proj_matrix) @ ray_clip
        # ray_eye /= ray_eye[3]
        ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])
        
        view_matrix = np.array(self.viewMatrix().data()).reshape(4, 4).T
        ray_world = np.linalg.inv(view_matrix) @ ray_eye
        ray_world = ray_world[:3] / np.linalg.norm(ray_world[:3])
        
        return ray_world
    
    def calculate_intersection(self, ray):
        # Get the camera position and direction
        camera_pos = np.array(self.cameraPosition())
        camera_dir = self.opts['center'] - camera_pos
        camera_dir = camera_dir / np.linalg.norm(camera_dir)
        
        # Calculate the plane normal (perpendicular to camera direction)
        plane_normal = camera_dir
        
        # Calculate the distance from the camera to the center of the view
        d = np.dot(self.opts['center'] - camera_pos, plane_normal)
        
        # Calculate the intersection with the plane
        denom = np.dot(ray, plane_normal)
        if abs(denom) > 1e-6:
            t = d / denom
            if t >= 0:
                intersection = camera_pos + t * ray
                return intersection
        
        return None
    
    def draw_ray(self, intersection):
        camera_pos = np.array(self.cameraPosition())
        
        # Create line data
        line_data = np.array([camera_pos, intersection])
        self._ray.setData(pos=line_data, color=(1, 0, 0, 1), width=2)

class GLSelectableScatterViewWidget(gl.GLViewWidget):
    sel_eps: float=np.inf

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sp = gl.GLScatterPlotItem(pxMode=True)
        self._sp.setGLOptions('translucent')
        self._sp_sel = gl.GLScatterPlotItem(pxMode=True)
        self.addItem(self._sp)
        self.addItem(self._sp_sel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # State
        self._pts = None
        self._proj_pts = None
        self._selected = np.array([], dtype=np.intp)
        self._hovered = None

    def setScatterData(self, pts: np.ndarray, *args, **kwargs):
        self._pts = pts
        self._sp.setData(pos=self._pts, *args, **kwargs)

    def _project(self):
        if not self._pts is None:
            print('ran projection')
            # Get the current view and projection matrices
            view_matrix = np.array(self.viewMatrix().data()).reshape(4, 4)
            proj_matrix = np.array(self.projectionMatrix(region=None).data()).reshape(4, 4)

            # Combine view and projection matrices
            mvp_matrix = np.dot(proj_matrix, view_matrix)

            # Project 3D points to normalized device coordinates
            points_4d = np.column_stack((self._pts, np.ones(self._pts.shape[0])))
            projected_points = np.dot(mvp_matrix, points_4d.T).T

            # Perform perspective division
            projected_points[:, :3] /= projected_points[:, 3, None]

            # Convert to viewport coordinates
            _, __, width, height = self.getViewport()
            self._proj_pts = np.column_stack((
                (projected_points[:, 0] + 1) * width / 2,
                (1 - projected_points[:, 1]) * height / 2
            ))

    def paintGL(self, *args, **kwargs):
        super().paintGL(*args, **kwargs)
        self._project()

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if not self._proj_pts is None:
            pos = ev.pos()
            dists = np.linalg.norm(self._proj_pts - np.array([pos.x(), pos.y()]), axis=1)
            idx = np.argmin(dists)
            if dists[idx] < self.sel_eps:
                hovered = idx
            else:
                hovered = None
            print(f'hovered: {self._hovered}')
            if hovered != self._hovered:
                self._hovered = hovered
                self._redraw_sel()

    def _redraw_sel(self):
        if self._hovered is None:
            self._sp_sel.setData(pos=np.array([]))
        else:
            self._sp_sel.setData(pos=self._proj_pts[[self._hovered]], size=40)

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

