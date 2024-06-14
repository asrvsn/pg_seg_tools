'''
Pyqtgraph OpenGL widgets
'''
from typing import List
from qtpy.QtCore import Qt
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

class GLSyncedCameraViewWidget(GrabbableGLViewWidget):
    '''
    Shamelessly taken from 
    https://stackoverflow.com/questions/70551355/link-cameras-positions-of-two-3d-subplots-in-pyqtgraph
    '''

    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler'):
        self.linked_views: List[GrabbableGLViewWidget] = []
        super().__init__(parent, devicePixelRatio, rotationMethod)

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
