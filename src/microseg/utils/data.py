'''
Utils for reading data and metadata
'''

import numpy as np
from aicsimageio import AICSImage

def get_voxel_size(path: str) -> np.ndarray:
    '''
    Get physical pixel sizes
    '''
    aimg = AICSImage(path)
    return np.asarray(aimg.physical_pixel_sizes)