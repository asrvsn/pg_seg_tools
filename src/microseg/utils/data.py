'''
Utils for reading data and metadata
'''

import numpy as np
import os
from aicsimageio import AICSImage
from PIL import Image
from typing import Tuple, Optional
from skimage.io import imread
import pdb
import pickle
import sys

def get_voxel_size(path: str) -> np.ndarray:
    '''
    Get physical pixel sizes
    '''
    aimg = AICSImage(path)
    return np.asarray(aimg.physical_pixel_sizes)

def load_XY_image(path: str, gray: bool=True, imscale: Optional[Tuple[float, float]]=None) -> np.ndarray:
    '''
    Return an XY image.
    gray: if True, convert to grayscale
    imscale: optional rescaling factor for images
    '''
    assert os.path.exists(path), f'File not found: {path}'
    _, fext = os.path.splitext(path)
    if fext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        im = Image.open(path)
        if not imscale is None:
            # Rescale x/y axes 
            r1, r2 = round(im.size[0] * imscale[0]), round(im.size[1] * imscale[1])
            print(f'Rescaling image from {im.size} to {(r1, r2)}')
            im = im.resize((r1, r2))
        if gray and im.mode in ['RGB', 'RGBA']:
            im = im.convert('L')
        img = np.array(im)
    else:
        raise NotImplementedError(f'File type {fext} not supported')
    if gray:
        assert img.ndim == 2, f'Expected 2d image, got {img.ndim}d image'
    return img

def load_stack(path: str, imscale: Optional[Tuple[float, float]]=None) -> np.ndarray:
    '''
    Return data in ZXYC format
    imscale: optional rescaling factor for slices
    '''
    assert os.path.exists(path), f'File not found: {path}'
    _, fext = os.path.splitext(path)
    if fext == '.czi':
        img = AICSImage(path).get_image_data('ZXYC') # ?
        assert imscale is None, 'Rescaling not supported for CZI files yet'
    elif fext in ['.tif', '.tiff']:
        img = imread(path)
        if img.ndim == 3:
            assert img.shape[2] < img.shape[0], f'Expected XYC image, got too many channels: {img.shape}'
            img = np.array([img])
        if not imscale is None:
            # Resize each slice (now in ZXYC)
            r1, r2 = round(img.shape[1] * imscale[0]), round(img.shape[2] * imscale[1])
            print(f'Rescaling image from {img.shape[1:3]} to {(r1, r2)}')
            # Restack to ZCXY for rescaling
            img = img.transpose(0, 3, 1, 2)
            img = np.array([
                np.array([
                    np.array(Image.fromarray(chan).resize((r1, r2)))
                    for chan in frame
                ])
                for frame in img
            ])
            # Restack to ZXYC
            img = img.transpose(0, 2, 3, 1)
    elif fext in ['.seg']: # Support Segmentation2D files, uses the z-projected version
        # Monkey-patch modules
        from microseg.data import seg_2d
        from matgeo import plane
        sys.modules['seg_2d'] = seg_2d 
        sys.modules['plane'] = plane
        seg = pickle.load(open(path, 'rb'))
        img = seg.zproj.copy()
        # Reshape CXY to ZXYC
        img = np.array([img.transpose(1, 2, 0)])
        return img
    elif fext in ['.jpg', '.jpeg', '.png']:
        img = load_XY_image(path, imscale=imscale, gray=False)
        img = np.array([img])
    else:
        raise NotImplementedError(f'File type {fext} not supported')
    # data = itk.array_view_from_image(itk.imread(args.data))
    assert img.ndim == 4, f'Expected 4d image, got {img.ndim}d image'
    return img