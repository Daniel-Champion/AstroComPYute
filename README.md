# AstroComPYute
AstroComPYute is a python based, CUDA accelerated, computational astrophotography toolkit capable of end-to-end workflows and near-real-time processing.

Prerequisites:


  -- Nvidia GPU that supoorts the CUDA Toolkit 11.8 or 12+
  
  -- Nvidia CUDA Toolkit
  
  -- NASA Deep Star Map .exr file 
      (obtain from https://svs.gsfc.nasa.gov/4851)
      AstroComPYute uses the Deep Star Map to perform 
      automatic gradient removal in stacked astro-images.
      
  -- H-alpha full sky map .fits file from WHAM + VTSS + SHASSA all sky survey
      (https://faun.rc.fas.harvard.edu/dfink/skymaps/halpha/data/v1_1/index.html)
      AstroComPYute uses the Deep Star Map to proform automatic gradient removal 
      in stacked astro-images.
  
 -- Starnet++ v2.0.0 command line tool version of Starnet++
      (obtain from: https://www.starnetastro.com/)
  
 -- Python: cupy, opencv-python, scikit-image, astropy, raypy, numpy, scipy, matplotlib
  
  ...and your favorite data acquisition software



AstroComPYute has been tested with ZWO ASI one shot color (OSC) astronomy cameras (.tif and .fits) and late model Canon DSLR cameras (that produce .cr3 raw files).  AstroComPYute needs your help testing on other imagers and determinging color deconvolution matrices and debayer settings to work seamlessly.  


