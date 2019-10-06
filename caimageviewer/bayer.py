'''
Software bayer demosaic, used only in the non-OpenGL viewer widget

Adapted from colour-science/colour-demosaicing available at:
    https://github.com/colour-science/

(See license below)
'''
#
# Original license is as follows:
#   Copyright (c) 2015-2018, Colour Developers
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright
#         notice, this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimer in the
#         documentation and/or other materials provided with the distribution.
#       * Neither the name of the Colour Developers nor the
#         names of its contributors may be used to endorse or promote products
#         derived from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL COLOUR DEVELOPERS BE LIABLE FOR ANY
#   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#   DAMAGE.


import numpy as np

try:
    from scipy.ndimage.filters import convolve
except ImportError:
    def convolve(array, filter):
        '''2D convolution using only numpy

        Reference: https://stackoverflow.com/a/43087771
        '''
        s = filter.shape + tuple(np.subtract(array.shape, filter.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        sub_m = strd(array, shape=s, strides=array.strides * 2)
        return np.einsum('ij,ijkl->kl', filter, sub_m)


def masks(shape, pattern='RGGB'):
    """
    Returns the *Bayer* CFA red, green and blue masks for given pattern.
    Parameters
    ----------
    shape : array_like
        Dimensions of the *Bayer* CFA.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    tuple
        *Bayer* CFA red, green and blue masks.

    Examples
    --------
    >>> from pprint import pprint
    >>> shape = (3, 3)
    >>> pprint(masks(shape))
    (array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool))
    >>> pprint(masks(shape, 'BGGR'))
    (array([[False, False, False],
           [False,  True, False],
           [False, False, False]], dtype=bool),
     array([[False,  True, False],
           [ True, False,  True],
           [False,  True, False]], dtype=bool),
     array([[ True, False,  True],
           [False, False, False],
           [ True, False,  True]], dtype=bool))
    """

    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


def mosaic(RGB, pattern='RGGB'):
    """
    Returns the *Bayer* CFA mosaic for a given *RGB* colourspace array.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.

    Returns
    -------
    ndarray
        *Bayer* CFA mosaic.

    Examples
    --------
    >>> RGB = np.array([[[0, 1, 2],
    ...                  [0, 1, 2]],
    ...                 [[0, 1, 2],
    ...                  [0, 1, 2]]])
    >>> mosaic(RGB)
    array([[0, 1],
           [1, 2]])
    >>> mosaic(RGB, pattern='BGGR')
    array([[2, 1],
           [1, 0]])
    """

    RGB = np.asarray(RGB)
    R, G, B = RGB[..., 0], RGB[..., 1], RGB[..., 2]
    R_m, G_m, B_m = masks(RGB.shape[0:2], pattern)
    CFA = R * R_m + G * G_m + B * B_m
    return CFA


def demosaic(cfa, pattern='RGGB'):
    """
    Returns the demosaiced *RGB* colourspace array from given *Bayer* CFA using
    bilinear interpolation.

    Parameters
    ----------
    CFA : array_like
        *Bayer* color filter array (CFA).
    pattern : unicode, optional
        **{'RGGB', 'BGGR', 'GRBG', 'GBRG'}**,
        Arrangement of the colour filters on the pixel array.
    Returns
    -------
    ndarray
        *RGB* colourspace array.
    Notes
    -----
    -   The definition output is not clipped in range [0, 1] : this allows for
        direct HDRI / radiance image generation on *Bayer* CFA data and post
        demosaicing of the high dynamic range data as showcased in this
        `Jupyter Notebook <https://github.com/colour-science/colour-hdri/\
blob/develop/colour_hdri/examples/\
examples_merge_from_raw_files_with_post_demosaicing.ipynb>`_.
    References
    ----------
    -   :cite:`Losson2010c`
    Examples
    --------
    >>> CFA = np.array(
    ...     [[0.30980393, 0.36078432, 0.30588236, 0.3764706],
    ...      [0.35686275, 0.39607844, 0.36078432, 0.40000001]])
    >>> demosaic(CFA)
    array([[[ 0.69705884,  0.17941177,  0.09901961],
            [ 0.46176472,  0.4509804 ,  0.19803922],
            [ 0.45882354,  0.27450981,  0.19901961],
            [ 0.22941177,  0.5647059 ,  0.30000001]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.29705883],
            [ 0.15392157,  0.26960785,  0.59411766],
            [ 0.15294118,  0.4509804 ,  0.59705884],
            [ 0.07647059,  0.18431373,  0.90000002]]])
    >>> CFA = np.array(
    ...     [[0.3764706, 0.360784320, 0.40784314, 0.3764706],
    ...      [0.35686275, 0.30980393, 0.36078432, 0.29803923]])
    >>> demosaic(CFA, 'BGGR')
    array([[[ 0.07745098,  0.17941177,  0.84705885],
            [ 0.15490197,  0.4509804 ,  0.5882353 ],
            [ 0.15196079,  0.27450981,  0.61176471],
            [ 0.22352942,  0.5647059 ,  0.30588235]],
    <BLANKLINE>
           [[ 0.23235295,  0.53529412,  0.28235295],
            [ 0.4647059 ,  0.26960785,  0.19607843],
            [ 0.45588237,  0.4509804 ,  0.20392157],
            [ 0.67058827,  0.18431373,  0.10196078]]])
    """

    cfa = np.asarray(cfa)
    R_m, G_m, B_m = masks(cfa.shape, pattern)

    H_G = np.asarray([[0, 1, 0],
                      [1, 4, 1],
                      [0, 1, 0]]) / 4

    H_RB = np.asarray([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]]) / 4

    R = convolve(cfa * R_m, H_RB)
    G = convolve(cfa * G_m, H_G)
    B = convolve(cfa * B_m, H_RB)
    return np.concatenate(
        [R[..., np.newaxis], G[..., np.newaxis], B[..., np.newaxis]],
        axis=-1
    )
