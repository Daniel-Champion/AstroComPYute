#!/usr/bin/env python3
# Copyright: Multiple Authors
#
# This file is part of AstroComPYute. https://github.com/Daniel-Champion/AstroComPYute
#
# SPDX-License-Identifier: MIT

"""tests for file input/output"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
from skimage import data, io

# FIXME: Remove when project is installable
sys.path.append(str(Path(__file__).parent.parent))

from astrocompyute import fileio


class TestIO(unittest.TestCase):
    def setUp(self):
        """Create a directory with some valid files"""
        self.moon = data.moon()  # 2D uint8

    def test_fits(self):
        """ensure fits OK"""
        tmp_img = tempfile.NamedTemporaryFile(suffix=".fits")
        _ = fits.PrimaryHDU(self.moon).writeto(tmp_img.name)
        img_rgb, img_bayer = fileio.ReadASI_TIF_FITS(tmp_img.name)
        self.assertEqual(self.moon.shape, img_bayer.shape)
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertEqual(img_rgb.dtype, np.dtype(np.float64))

    def test_tif(self):
        """ensure tif OK"""
        tmp_img = tempfile.NamedTemporaryFile(suffix=".tif")
        io.imsave(tmp_img.name, self.moon)
        img_rgb, img_bayer = fileio.ReadASI_TIF_FITS(tmp_img.name)
        self.assertEqual(self.moon.shape, img_bayer.shape)
        self.assertEqual(img_rgb.ndim, 3)
        self.assertEqual(img_rgb.shape[2], 3)
        self.assertEqual(img_rgb.dtype, np.dtype(np.float64))

    def test_cr(self):
        """ensure cr2/cr3 compatibility"""
        self.skipTest("TODO: Write test for Canon images. No valid writer known.")

if __name__ == "__main__":
    unittest.main()
