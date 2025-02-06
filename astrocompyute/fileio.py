# -*- coding: utf-8 -*-
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import skimage.io as skio
from astropy.io import fits

from astrocompyute.imagemath import DownsampleImage


def ReadASI_TIF_FITS(
    asi_tif_fp: str,
    salt: bool = True,
    scalarRGB: bool = False,
    Red_Corr: float = 1.7566594340058168,
    Green_Corr: float = 1.0,
    Blue_Corr: float = 1.664086416513601,
) -> (np.ndarray, np.ndarray):
    """
    Read arbitrary TIF/CR2/CR3/FITS format as numpy object.

    Parameters
    ----------
    asi_tif_fp : str
        File path for ASI formatted image with BGGR bayer pattern.
    salt : bool, default True
        When converting to float add epsilon to extended precision.
    scalarRGB : bool, default False
        Return scaled & clipped image

    Returns
    -------
    image_rgb : ndarray of np.float64
    image_bayer : ndarray of np.uint8 or None

    Raises
    ------
    ValueError
        If strangely formatted image file.
    """
    raw = Path(asi_tif_fp)

    if raw.suffix.lower() in [".tif", ".tiff"]:
        image_bayer = skio.imread(asi_tif_fp)
        image_rgb16 = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_RG2BGR)

    elif raw.suffix.lower() in [".cr2", ".cr3"]:
        rawIM16 = rawpy.imread(asi_tif_fp)
        image_rgb16 = rawIM16.postprocess(
            # gamma=(1, 1),  # default is (2.222, 4.5)
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            # use_auto_wb = True,
            exp_shift=1.0,
            exp_preserve_highlights=1.0,
        )
        image_bayer = None

    else:
        with fits.open(asi_tif_fp) as hdul:
            image_bayer = deepcopy(hdul[0].data)
        image_rgb16 = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_RG2BGR)

    # sanity check
    if image_bayer.ndim != 2:
        raise ValueError(f"image is strange shape: {image_bayer.shape}")
    if image_bayer.dtype != np.uint8:
        raise ValueError(f"unexpected image datatype {image_bayer.dtype}")
    # type conversion & scaling
    image_rgb = image_rgb16.astype(np.float64) / 2**16
    if salt:
        # add variation to uint8 measurements
        image_rgb += np.random.random(image_rgb.shape) / 2**16
    # optional convertion
    if scalarRGB:
        image_rgb *= [Red_Corr, Green_Corr, Blue_Corr]
        image_rgb = np.clip(image_rgb, a_min=None, a_max=1)

    return image_rgb, image_bayer


def SaveImage2Disk(
    SaveImage: np.ndarray,
    OutputDir: str = ".",
    starless_version=None,
    description: str = "",
    save_token: str = "",
) -> None:
    """
    Save a processed image as a variety of formats.

    Parameters
    ----------
    SaveImage : np.ndarray
        Normalized float image usually scaled 0-1 of shape HWC.
    OutputDir : str | bytes | PathLike
        Directory to save files.
    """
    output_dir = Path(OutputDir)
    SaveImage[np.isnan(SaveImage)] = 0
    max_value = (2**16 - 1) / 2**16
    # Create a 2x2 binned smaller resolution image
    SaveImage_2x2 = DownsampleImage(
        SaveImage,
        bin_size=2,
        resolution=(int(SaveImage.shape[0] / 2), int(SaveImage.shape[1] / 2)),
    )

    ## create clipped original
    out_im = np.clip(SaveImage, 0, max_value)

    ## create clipped downsample
    out_im_2x2 = np.clip(SaveImage_2x2, 0, max_value)

    if starless_version is not None:
        out_im_starless = np.clip(starless_version, 0, max_value)

    # save 2x2 binned
    plt.imsave(output_dir / f"{description}_{save_token}_plt_2x2.png", out_im_2x2)
    plt.imsave(output_dir / f"{description}_{save_token}_plt_2x2.jpg", out_im_2x2)
    # save full resolution
    plt.imsave(output_dir / f"{description}_{save_token}_plt.png", out_im)
    plt.imsave(output_dir / f"{description}_{save_token}_plt.jpg", out_im)
    # save the starless if provided
    if starless_version:
        plt.imsave(
            output_dir / f"{description}_{save_token}_plt_starless.png", out_im_starless
        )
        plt.imsave(
            output_dir / f"{description}_{save_token}_plt_starless.jpg", out_im_starless
        )

    # OpenCV can do a high quality 16-bit tif save, but we need to swap red and blue channels
    img_swap = np.zeros_like(out_im)
    img_swap[:, :, 0] = out_im[:, :, 2]
    img_swap[:, :, 1] = out_im[:, :, 1]
    img_swap[:, :, 2] = out_im[:, :, 0]
    img_uint16 = np.round(img_swap * 2**16).astype(np.uint16)
    cv2.imwrite(output_dir / f"{description}_{save_token}.tif", img_uint16)
