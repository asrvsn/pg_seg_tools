''' Colors '''

import numpy as np
import colorcet as cc
import seaborn as sns
import colorsys
import pdb
from typing import Any

''' Categorical '''

cc_glasbey_01 = np.array(
    # [np.array([0,0,0])] + # First color is black
    [np.array(rgb) for rgb in cc.glasbey_bw_minc_20_minl_30] + 
    [np.array([1,1,1])], # Last color is white
    dtype=np.float32
)

cc_glasbey_01_rgba = np.concatenate([cc_glasbey_01, np.ones((cc_glasbey_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_glasbey_dark_01 = np.array(
    # [np.array([0,0,0])] + # First color is black
    [np.array(rgb) for rgb in cc.glasbey_bw_minc_20] + 
    [np.array([1,1,1])], # Last color is white
    dtype=np.float32
)

cc_glasbey_255 = (cc_glasbey_01 * 255).astype(np.uint8)

cc_glasbey_255_rgba = (cc_glasbey_01_rgba * 255).astype(np.uint8)

''' Continuous '''


cc_fire_01 = np.array(cc.linear_kryw_0_100_c71)
cc_fire_01_rgba = np.concatenate([cc_fire_01, np.ones((cc_fire_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_cet_01 = np.array(cc.isoluminant_cgo_70_c39)
cc_cet_01_rgba = np.concatenate([cc_cet_01, np.ones((cc_cet_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_bmy_01 = np.array(cc.linear_bmy_10_95_c78)
cc_bmy_01_rgba = np.concatenate([cc_bmy_01, np.ones((cc_bmy_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_kgy_01 = np.array(cc.linear_kgy_5_95_c69)
cc_kgy_01_rgba = np.concatenate([cc_kgy_01, np.ones((cc_kgy_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_bgy_01 = np.array(cc.linear_bgy_10_95_c74)
cc_bgy_01_rgba = np.concatenate([cc_bgy_01, np.ones((cc_bgy_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_blues_01 = np.array(cc.linear_blue_95_50_c20)
cc_blues_01_rgba = np.concatenate([cc_blues_01, np.ones((cc_blues_01.shape[0], 1), dtype=np.float32)], axis=1)

# General continuous maps
cc_cont = {
    'fire': {
        '01': cc_fire_01,
        '01_rgba': cc_fire_01_rgba
    },
    'cet': {
        '01': cc_cet_01,
        '01_rgba': cc_cet_01_rgba
    },
    'bmy': {
        '01': cc_bmy_01,
        '01_rgba': cc_bmy_01_rgba
    },
    'kgy': {
        '01': cc_kgy_01,
        '01_rgba': cc_kgy_01_rgba
    },
    'bgy': {
        '01': cc_bgy_01,
        '01_rgba': cc_bgy_01_rgba
    },
    'blues': {
        '01': cc_blues_01,
        '01_rgba': cc_blues_01_rgba
    },
}

''' Polar '''

cc_cet_c11_01 = np.array(cc.cyclic_bgrmb_35_70_c75, dtype=np.float32)
cc_cet_c11_01_rgba = np.concatenate([cc_cet_c11_01, np.ones((cc_cet_c11_01.shape[0], 1), dtype=np.float32)], axis=1)

cc_cet_c11s_01 = np.array(cc.cyclic_bgrmb_35_70_c75_s25, dtype=np.float32)
cc_cet_c11s_01_rgba = np.concatenate([cc_cet_c11s_01, np.ones((cc_cet_c11s_01.shape[0], 1), dtype=np.float32)], axis=1)

''' General utilities '''

def map_colors(
        data: np.ndarray, 
        scheme='categorical', 
        cont_map='fire',
        rgba: bool=False, 
        i255: bool=False, 
        alpha: float=1., 
        d_min: float=None, 
        d_max: float=None,
    ) -> np.ndarray:
    '''
    Map data into a color space using a coloring scheme.
    Supported coloring schemes: 
    - 'categorical': map integer-valued data into binned color space
    - 'mask': map integer-valued data into binned rgba 255 color space, where 0 is transparent
    - 'continuous': map float-valued data into real-valued color space, implying there is a color-preferential ordering (heatmap)
    - 'continuous_fixed': map float-valued data into a fixed color space, implying there is no color-preferential ordering (diverging)
    - 'periodic': map data into periodic color space 
    If rgba=True, use 4-channel color space (RGBA), otherwise use 3-channel color space (RGB).
    '''
    def check_bounds():
        assert not d_min is None and not d_max is None, 'Must specify d_min and d_max for continuous_fixed scheme'
        assert not np.isclose(d_min, d_max), 'd_min and d_max must be different'
        assert d_min < d_max, 'd_min must be less than d_max'
        assert (d_min <= data).all() and (data <= d_max).all(), 'data must be in range [d_min, d_max]'
    if scheme == 'categorical':
        assert data.dtype == int
        colors = cc_glasbey_01_rgba if rgba else cc_glasbey_01
        colors = (colors * 255).astype(np.uint8) if i255 else colors
        colors = colors[1:]
        return colors[data % colors.shape[0]]
    elif scheme == 'mask':
        assert data.ndim == 2, 'mask must be 2D'
        colors = cc_glasbey_255_rgba if i255 else cc_glasbey_01_rgba
        colors = colors[data % colors.shape[0]]
        colors[:, :, 3] = int(alpha * 255)
        colors[data == 0] = 0
        return colors
    elif scheme == 'continuous':
        colors = cc_cont[cont_map]
        colors = colors['01_rgba'] if rgba else colors['01']
        colors = (colors * 255).astype(np.uint8) if i255 else colors
        d_min, d_max = data.min(), data.max()
        data = (data - d_min) / (d_max - d_min) if not np.isclose(d_min, d_max) else np.full_like(data, 0.5) # Map to [0, 1]
        data = np.round(data * (colors.shape[0] - 1)).astype(int) # Map to [0, n_colors)
        colors = colors[data]
        if rgba:
            colors[:, 3] = alpha
        return colors
    elif scheme == 'continuous_fixed':
        colors = cc_cont[cont_map]
        colors = colors['01_rgba'] if rgba else colors['01']
        colors = (colors * 255).astype(np.uint8) if i255 else colors
        check_bounds()
        data = (data - d_min) / (d_max - d_min) # Map to [0, 1]
        data = np.round(data * (colors.shape[0] - 1)).astype(int) # Map to [0, n_colors)
        return colors[data]
    elif scheme == 'periodic':
        cmap = sns.husl_palette(as_cmap=True)
        check_bounds()
        data = (data - d_min) / (d_max - d_min) # Map to [0, 1]
        colors = cmap(data)
        colors = (colors * 255).astype(np.uint8) if i255 else colors
        return colors
    else:
        raise ValueError(f'Unsupported color scheme: {scheme}')

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def generate_max_contrast_colors(n):
    assert n > 0, 'n must be greater than 0'
    # Generate equally spaced hues
    hues = [i / n for i in range(n)]
    # Convert hues to RGB colors
    rgb_colors = [colorsys.hls_to_rgb(hue, 0.5, 0.9) for hue in hues]
    # Convert RGB colors to 8-bit values
    max_contrast_colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in rgb_colors]
    # Convert RGB colors to hexadecimal format
    hex_colors = [rgb_to_hex(rgb) for rgb in max_contrast_colors]
    return hex_colors

def map_color_01(val: float, wheel: np.ndarray) -> Any:
    assert 0 <= val <= 1
    n = len(wheel)
    return wheel[round(val * (n-1))]