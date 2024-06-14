from typing import Tuple, Optional
import numpy as np

from matgeo.plane import *

AVAIL_ZPROJ_MODES = ['slice', 'max', 'min', 'mean', 'median']
AVAIL_CP_MODELS = ['cyto2', 'cyto', 'nuclei']
AVAIL_DENOISE_MODES = ['None', 'Bi', 'TV_B', 'Wvt'] # See scikit-image restoration API

class Segmentation2DSettings:
    '''
    Settings for 2D segmentation
    - z projection method
    - cellpose segmentation settings 
    - manual segmentation settings
    '''
    def __init__(self, czxy: Tuple[int, int, int, int]):
        # Applies globally
        self.czxy = czxy
        c, z, x, y = czxy
        # Applies per channel
        self.zproj_mode = [AVAIL_ZPROJ_MODES[0] for _ in range(c)]
        self.zproj_window = [(z//2, z//2) for _ in range(c)] # Inclusive
        self.cp_denoise = [AVAIL_DENOISE_MODES[0] for _ in range(c)]
        self.cp_model = [AVAIL_CP_MODELS[0] for _ in range(c)]
        self.cp_nuclear_channel = [None for _ in range(c)]
        self.cp_diam = [30 for _ in range(c)]
        self.cp_flow = [0.4 for _ in range(c)] # Flow thresholds
        self.cp_cellprob = [0.0 for _ in range(c)] # Cell probability thresholds

class Segmentation2D:
    ''' 
    2D Segmentation of an CZXY stack 
    - czxy: shape of the data in CZXY format
    - upp: units per pixel (XY)
    '''
    def __init__(self, czxy: Tuple[int, int, int], upp: Tuple[float, float]):
        self.czxy = czxy
        self.upp = upp
        self.settings = Segmentation2DSettings(czxy)
        cxy = (czxy[0], czxy[2], czxy[3])
        # Computed z-projection
        self.zproj: np.ndarray = np.zeros(cxy, dtype=np.uint16)
        # CXY image inputs for segmentation (after z-projection, including any denoising, etc.)
        self.img: np.ndarray = self.zproj.copy()
        # Outline as a polygon (in XY coordinates)
        self.outline: Optional[PlanarPolygon] = None
        # Mask of selected regions, CXY (zero for background), computed by cellpose
        self.cp_mask: np.ndarray = np.zeros(cxy, dtype=np.uint32) # Needs to be uint32 for upolygon
        # Mask after manual curation
        self.mask: np.ndarray = self.cp_mask.copy()

    @property
    def circular_radius(self) -> Optional[float]:
        ''' Radius of the circular region in um '''
        if not self.outline is None:
            poly = self.outline.set_res(*self.upp)
            return poly.circular_radius()