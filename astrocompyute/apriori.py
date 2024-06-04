# -*- coding: utf-8 -*-
"""
Created on Sat May 25 09:57:53 2024

@author: champ
"""


import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import time

from copy import deepcopy

import numpy as np
import scipy as sp

import numpy.random as npr
import cv2

import skimage.io as skio

import cupy as cp
#import cupyx as cpx

from astrocompyute.fileio import ReadASI_TIF_FITS
from astrocompyute.color import RGB2HSV
from astrocompyute.filters import maximumDiskFilter, maximumBoxFilterRC
from astrocompyute.imagemath import GetImageCoords
import astrocompyute.fileio as fileio
from astropy.io import fits

from astropy import units as u
from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt

###########################################
#### General coordinate transform methods


def degDMS(degreez):
    """Converts floating point degrees to Degrees Minutes Seconds.
    
    Returns
    -------
        DMS : tuple (int, int, float)
        
    """
    ddd = int(abs(degreez))
    if degreez < 0:
        dsign = -1
    else:
        dsign = 1
    mmm = int( 60.*(abs(degreez) - ddd)  )
    sss = 3600. * (abs(degreez) - ddd - mmm/60.)
    return dsign*ddd, mmm, sss

def DMSdeg(dms):
    """Converts Degrees Minutes Seconds to float degrees.
    
    Returns
    -------
        degrees : float
        
    """
    if dms[0] < 0:
        dsign = -1
    else:
        dsign = 1
        
    deg = abs(dms[0]) + dms[1]/60. + dms[2]/3600. 
    return dsign * deg
    
def HMSdeg(hms):
    """Converts Hours Minutes Seconds tuple to float degrees.
    
    Returns
    -------
        degrees : float
        
    """
    if hms[0] < 0:
        dsign = -1
    else:
        dsign = 1    
    
    deg = 360.*( abs(hms[0])/24. + hms[1]/(24.*60) + hms[2] / (24*60*60)   )
    
    return dsign * deg
    
def degHMS(degreez):
    """Converts floating point degrees to Hours Minutes Seconds.
    
    Returns
    -------
        HMS : tuple (int, int, float)
        
    """
    hhh_float = 24./360 * degreez
    hhh = int(hhh_float)
    mmm = int( 60.*(hhh_float - hhh)  )
    sss = 3600. * (hhh_float - hhh - mmm/60.)
    return hhh, mmm, sss


###########################################
#### ASTAP Plate Solving Wrappers


def PlateSolve(sample_image_path, 
               ra_guess = None, dec_guess = None, 
               fov = 0 , 
               search = 140, 
               astap_executable = '"C:/Program Files/astap/astap_cli.exe"'): # fov = 8.55 for sigma 105
    """Python wrapper for the ASTAP plate solving CLI.

    Provide this function a path to an image an it will call ASTAP's CLI to 
    perform a plate solve of that image.  Parses output from ASTAP and returns
    a Python dictionary of plate solve output. 

    Parameters
    ----------
        sample_image_path : string
            full file path to image file
        ra_guess
            Guess of RA coordinate (optional)
        dec_guess
            Guess of DEC coordinate (optional)
        fov
            Guess of imager field of view in degrees (optional)
        search
            degree span for search (optional)
        astap_executable
            full file path to ASTAP CLI executable (the executable named astap_cli.exe)

    Returns
    -------
        solution : dictionary
            plate solve solution stored in a dictionary.  The most critical 
            entities being:
                the center pixel plate solve coordinates = solution['ref_RADEC']
                The CD matrix (see ASTAP website) that can assist in determinging 
                the tangent plane coordinates of each pixel = solution['CD_matrix']

    """
    T1 = time.time()
    sample_image_dir, sample_image_fn = os.path.split(sample_image_path)
    
    if sample_image_fn[-4:] == '.cr3':
        kill_the_file = True
        new_fn = sample_image_fn.replace('.cr3', '.tif')
        
        temp_im, _imBayer0 = ReadASI_TIF_FITS(sample_image_path, salt = False)

        
        out_im = deepcopy(temp_im)
        
        out_im[:,:,0] = deepcopy(temp_im[:,:,2])
        out_im[:,:,2] = deepcopy(temp_im[:,:,0])
    
        cv2.imwrite(os.path.join(sample_image_dir, new_fn), np.array(out_im*2**16, dtype = np.uint16))
        sample_image_fn = new_fn
        _sample_image_path = os.path.join(sample_image_dir, new_fn)
    else:
        kill_the_file = False
        _sample_image_path = sample_image_path
    
    ini_fn = sample_image_fn.replace('.cr3', '').replace('.tif', '').replace('.fits', '') + '.ini'
    isSuccessOutput = os.path.join(sample_image_dir, ini_fn)
    isSuccessOutput_wcs = os.path.join(sample_image_dir, ini_fn.replace('.ini', '.wcs'))
    
    
    
    if type(ra_guess) != type(None):
        ra_guess = ' -ra ' + str(round(ra_guess, 3)) 
    else:
        ra_guess = ''
    if type(dec_guess) != type(None):
        dec_guess = ' -spd ' +str( round(dec_guess + 90, 3) )
    else:
        dec_guess = ''
        
        
    ASTAP_command = astap_executable + ' -f ' +  _sample_image_path + ' -r '+ str(int(search)) + ' -fov ' + str(fov) + ra_guess + dec_guess
    
    os.system(ASTAP_command)
    
    
    if ini_fn in os.listdir(sample_image_dir):
        
        solution = open(isSuccessOutput, 'r').read()
        
        if 'PLTSOLVD=F' in solution:
            
            solution = None
        
        else:
            print('solution:\n', solution)
            solution = solution.split('\n')
            
            solution = { rr[:rr.index('=')]: rr[rr.index('=')+1:].strip() for rr in solution if (('=' in rr) and (len(rr) > 0))}
            
            
            for sk in solution:
                
                sk_val = solution[sk]
                
                try:
                    
                    sk_val = float(sk_val)
                    solution[sk] = sk_val
                except:
                    
                    pass
                
                
            CD_matrix = np.array([[solution['CD1_1'], solution['CD1_2']], [solution['CD2_1'], solution['CD2_2']]])
            
            ref_xy = np.array([solution['CRPIX1'], solution['CRPIX2']])
            
            ref_RADEC = np.array([solution['CRVAL1'], solution['CRVAL2']])
            
            solution['CD_matrix'] = CD_matrix
            solution['ref_xy'] = ref_xy
            solution['ref_RADEC'] = ref_RADEC
        
        try:
            os.remove(isSuccessOutput)
            os.remove(isSuccessOutput_wcs)
            if kill_the_file:
                os.remove(_sample_image_path)
        except:
            pass
        
    else:
        
        solution = None
        
    T2 = time.time()
    
    
    print('ASTAP Plate Solve in:', T2 - T1, 'seconds')    
    
    return solution


    
def PlateSolveMask(image_fp, 
                   astap_executable = '"C:/Program Files/astap/astap.exe"' ,
                   dilate_n = 200,
                   return_annotation_bool = False,
                   min_BG_pix_proportion = 0.5,
                   kill_the_file = True,
                   max_attempts = 10):
    """Perform a plate solve of an image and return an annotated version and a 
    boolean sampling mask that avoids annotated deep sky objects.

    Description of the function.

    Parameters
    ----------
        image_fp : string
            full file path to image file
        astap_executable : string
            full file path to base ASTAP executable (non CLI)  (the executable named astap.exe)
        dilate_n : int
            dilation amounbt in pixels for the boolean mask creation
        return_annotation_bool : bool
            set this to True to return the annotation in addition to the sampling boolean
        min_BG_pix_proportion : float between 0 and 1
            The background is identified as the component of the annotated image 
            that flood fills to at least this percentage of the image pixels. 
        kill_the_file : boolean
            delets the file created by ASTAP after the call. 
        max_attempts : int
            maximum number of flood fill attempts to find the background. 




    Returns
    -------
        not_DSO_mask : boolean ndarray
            sampling boolean array that avoids annotated DSOs
        annotation : boolean ndarray, if return_annotation_bool = True
            the boolean annotation array (same shape as the image)

    """    
    Annotate_command = astap_executable + ' -annotate -f ' + image_fp + ' -r 140 -fov 0 -check apply'
    
    os.system(Annotate_command)
    
    image_ext = image_fp.split('.')[-1]
    
    ps_annotated_fp = image_fp.replace('.' + image_ext, '_annotated.jpg')
    
    ps_annotated = skio.imread(ps_annotated_fp)
    
    ps_annotated_HSV = RGB2HSV(np.array(ps_annotated, dtype = float)/256.)
    
    
    
    annotate_bool = 1.0 * (ps_annotated_HSV[:,:,0] > 0.0)
    
    filled_pix = 0
    corners = [(0,0), 
               (annotate_bool.shape[0]-3,0), 
               (0,annotate_bool.shape[1]-4), 
               (annotate_bool.shape[0]-3,annotate_bool.shape[1]-4)]
    icorner = 0
    
    attempt = 0
    best_attempt = 0
    best_fill = None
    while (filled_pix < (min_BG_pix_proportion * (annotate_bool.shape[1]*annotate_bool.shape[0]))) and (attempt < max_attempts):
        annotate_bool_u8 = np.array(annotate_bool*255, dtype = np.uint8)
        fill_mask = np.zeros((annotate_bool.shape[0]+2, annotate_bool.shape[1]+2), np.uint8)
        
        th, im_th = cv2.threshold(annotate_bool_u8, 127, 255, cv2.THRESH_BINARY)
        if True: #icorner >=4:
            
            #seed_point = (npr.randint(4, annotate_bool.shape[0]-4), npr.randint(4, annotate_bool.shape[1]-4))
            seed_point = (npr.randint(4, annotate_bool.shape[1]-4), npr.randint(4, annotate_bool.shape[0]-4))
            print('qqq', seed_point)
            cv2.floodFill(annotate_bool_u8, 
                          fill_mask, 
                          seed_point, 
                          127)
        else:
            print('qqq', corners[icorner], annotate_bool_u8.shape, fill_mask.shape)
            cv2.floodFill(annotate_bool_u8, fill_mask, corners[icorner], 127)
        filled_pix = (annotate_bool_u8 == 127).sum()
        icorner += 1
        print('flood fill size=', filled_pix,round(100*filled_pix/(annotate_bool.shape[0] * annotate_bool.shape[1])), icorner)
        attempt += 1
        if filled_pix > best_attempt:
            best_attempt = filled_pix
            best_fill = deepcopy(annotate_bool_u8)
            
    annotate_bool_u8 = best_fill
            
    
    base_boolean = annotate_bool_u8 != 127
    #print((annotate_bool_u8 == 127).sum())
    
    filled_image = maximumBoxFilterRC(3, 2*dilate_n, np.array(( base_boolean ), dtype = np.uint8))
    filled_image = maximumBoxFilterRC(dilate_n, 1, filled_image)
    
    not_DSO_mask = ~(filled_image > 0)
    
    if kill_the_file:
        os.remove(ps_annotated_fp)
    
    if return_annotation_bool:
        yellow_proj = np.sum(ps_annotated/256. * np.array([[[0.7071067811865476, 0.7071067811865476, 0]]]), axis = 2)
        white_proj = np.sum(ps_annotated/256. , axis = 2)
        
        annotation = (yellow_proj > 1.0025) & (np.abs(ps_annotated_HSV[:,:,0] - 60) < 20) & (white_proj < 2.52)
        annotation = maximumDiskFilter(3, np.array(255*annotation, dtype = np.uint8))
        return not_DSO_mask, annotation
    
    return not_DSO_mask




def IWCTransform(pixel_rows, pixel_cols, platesolve_solution):
    
    CD_matrix = platesolve_solution['CD_matrix']
    
    ref_xy = np.array([platesolve_solution['CRPIX1'], platesolve_solution['CRPIX2']])

    ref_RADEC = np.array([platesolve_solution['CRVAL1'], platesolve_solution['CRVAL2']])


    px = pixel_cols
    py = pixel_rows 
    
    U =  (px - ref_xy[0])
    V =  -(py - ref_xy[1])
    
    iwcx = CD_matrix[0,0] * U + CD_matrix[0,1] * V + ref_RADEC[0]
    iwcy = CD_matrix[1,0] * U + CD_matrix[1,1] * V + ref_RADEC[1]
    
    return  iwcx, iwcy # RA and DEC on the tangent plane



def Pixels2RADEC_Transform(pixel_rows, pixel_cols, platesolve_solution):
    
    CD_matrix = platesolve_solution['CD_matrix']
    
    ref_xy = np.array([platesolve_solution['CRPIX1'], platesolve_solution['CRPIX2']])

    ref_RADEC = np.array([platesolve_solution['CRVAL1'], platesolve_solution['CRVAL2']])

    cosine_correction = np.cos(ref_RADEC[1] * np.pi / 180.)
    print('cosine_correction', cosine_correction)
    px = pixel_cols
    py = pixel_rows 
    
    U =  (px - ref_xy[0])
    V =  -(py - ref_xy[1])
    
    RA_coords = (CD_matrix[0,0] * U + CD_matrix[0,1] * V)/cosine_correction + ref_RADEC[0]
    DEC_coords = CD_matrix[1,0] * U + CD_matrix[1,1] * V + ref_RADEC[1]

    return  RA_coords, DEC_coords # RA and DEC on the tangent plane


def RADEC23D(RA, DEC):
    
    X_coords = np.cos(DEC * np.pi / 180.)*np.cos(RA * np.pi / 180.)
    Y_coords = np.cos(DEC * np.pi / 180.)*np.sin(RA * np.pi / 180.)
    Z_coords = np.sin(DEC * np.pi / 180.)
    
    if type(RA) == type(np.arange(3)):
        return np.vstack((X_coords, Y_coords, Z_coords)).T
        
    return np.array([X_coords, Y_coords, Z_coords])
    
    
def Pixels2RADEC_Transform3D(pixel_rows, pixel_cols, platesolve_solution, delta = 10):
    """Converts image pixel coordinates to pixel-specific RA DEC coordinates using 
    a plate solve solution.

    Using a plae solve solution, the coordinates on a tanget plane to the unit 
    celestial sphere are computed and then projected to the sphere.

    Parameters
    ----------
        pixel_rows : array_like (N,)
            the pixel row coordinates, int
        pixel_cols :  : array_like (N,)
            the pixel column coordinates, int
        platesolve_solution : dict
            plate solve solution from apriori.PlateSolve
        delta : int
            small pertubation amount that is used to numerically determine the 
            basis on the tangent plane. 10 works well.

    Returns
    -------
        coords3D_ra : array_like (N,)
            RA coordinates of each pixel
        coords3D_dec : array_like (N,)
            DEC coordinates of each pixel

    """
    dec_radians = platesolve_solution['ref_RADEC'][1] * np.pi / 180.
    ra_radians = platesolve_solution['ref_RADEC'][0] * np.pi / 180.
    
    TP3D = np.array([np.cos(dec_radians)*np.cos(ra_radians),
                     np.cos(dec_radians)*np.sin(ra_radians),
                     np.sin(dec_radians) ])
    
    delta_pixel_row = platesolve_solution['ref_xy'] + np.array([0, delta])
    delta_pixel_col = platesolve_solution['ref_xy'] + np.array([delta, 0])
    
    delta_pixel_row_RADEC = Pixels2RADEC_Transform(delta_pixel_row[1], delta_pixel_row[0], platesolve_solution)
    delta_pixel_col_RADEC = Pixels2RADEC_Transform(delta_pixel_col[1], delta_pixel_col[0], platesolve_solution)

    delta_pixel_row_3D = RADEC23D(delta_pixel_row_RADEC[0], delta_pixel_row_RADEC[1])
    delta_pixel_col_3D = RADEC23D(delta_pixel_col_RADEC[0], delta_pixel_col_RADEC[1])
    delta_pixel_row_3D  /= np.sum(delta_pixel_row_3D * TP3D)
    delta_pixel_col_3D  /= np.sum(delta_pixel_col_3D * TP3D)
    
    basis_row = (delta_pixel_row_3D - TP3D)/delta
    basis_col = (delta_pixel_col_3D - TP3D)/delta
    
    relative_row_coords = pixel_rows - platesolve_solution['ref_xy'][1]
    relative_col_coords = pixel_cols - platesolve_solution['ref_xy'][0]
    
    coords3D = basis_row.reshape(1,-1) * relative_row_coords.reshape(-1,1) 
    coords3D += basis_col.reshape(1,-1) * relative_col_coords.reshape(-1,1)
    coords3D += TP3D
    
    coords3D_rho = np.sqrt(np.sum(coords3D**2, axis = 1))
    
    coords3D_dec = np.arccos(coords3D[:,2] / coords3D_rho )
    coords3D_ra = np.sign(coords3D[:,1]) * np.arccos( coords3D[:,0] / np.sqrt( coords3D[:,0]**2 + coords3D[:,1]**2 ))
    
    coords3D_dec = 90. - (coords3D_dec * 180./np.pi)
    coords3D_ra *= 180./np.pi
    coords3D_ra += (coords3D_ra < 0)*360
    
    return  coords3D_ra, coords3D_dec # RA and DEC on the tangent plane





def Pixels2RADEC_Transform3D_cuda(pixel_rows, pixel_cols, platesolve_solution, delta = 10):
    """Converts image pixel coordinates to pixel-specific RA DEC coordinates using 
    a plate solve solution. CUDA version using cupy

    Using a plae solve solution, the coordinates on a tanget plane to the unit 
    celestial sphere are computed and then projected to the sphere.

    Parameters
    ----------
        pixel_rows : array_like (N,)
            the pixel row coordinates, int
        pixel_cols :  : array_like (N,)
            the pixel column coordinates, int
        platesolve_solution : dict
            plate solve solution from apriori.PlateSolve
        delta : int
            small pertubation amount that is used to numerically determine the 
            basis on the tangent plane. 10 works well.

    Returns
    -------
        coords3D_ra : array_like (N,)
            RA coordinates of each pixel
        coords3D_dec : array_like (N,)
            DEC coordinates of each pixel

    """
    dec_radians = platesolve_solution['ref_RADEC'][1] * np.pi / 180.
    ra_radians = platesolve_solution['ref_RADEC'][0] * np.pi / 180.
    
    TP3D = np.array([np.cos(dec_radians)*np.cos(ra_radians),
                     np.cos(dec_radians)*np.sin(ra_radians),
                     np.sin(dec_radians) ])
    
    delta_pixel_row = platesolve_solution['ref_xy'] + np.array([0, delta])
    delta_pixel_col = platesolve_solution['ref_xy'] + np.array([delta, 0])
    
    delta_pixel_row_RADEC = Pixels2RADEC_Transform(delta_pixel_row[1], delta_pixel_row[0], platesolve_solution)
    delta_pixel_col_RADEC = Pixels2RADEC_Transform(delta_pixel_col[1], delta_pixel_col[0], platesolve_solution)

    delta_pixel_row_3D = RADEC23D(delta_pixel_row_RADEC[0], delta_pixel_row_RADEC[1])
    delta_pixel_col_3D = RADEC23D(delta_pixel_col_RADEC[0], delta_pixel_col_RADEC[1])
    delta_pixel_row_3D  /= np.sum(delta_pixel_row_3D * TP3D)
    delta_pixel_col_3D  /= np.sum(delta_pixel_col_3D * TP3D)
    
    basis_row = (delta_pixel_row_3D - TP3D)/delta
    basis_col = (delta_pixel_col_3D - TP3D)/delta
    
    # # # #
    basis_row_gpu = cp.asarray(basis_row)
    basis_col_gpu = cp.asarray(basis_col)
    
    pixel_rows_gpu = cp.asarray(pixel_rows)
    pixel_cols_gpu = cp.asarray(pixel_cols)
    
    
    # cpu
    # relative_row_coords = pixel_rows - platesolve_solution['ref_xy'][1]
    # relative_col_coords = pixel_cols - platesolve_solution['ref_xy'][0]
    
    relative_row_coords = pixel_rows_gpu - float(platesolve_solution['ref_xy'][1])
    relative_col_coords = pixel_cols_gpu - float(platesolve_solution['ref_xy'][0])
    
    
    
    coords3D = basis_row_gpu.reshape(1,-1) * relative_row_coords.reshape(-1,1) 
    coords3D += basis_col_gpu.reshape(1,-1) * relative_col_coords.reshape(-1,1)
    coords3D += cp.asarray(TP3D)
    
    coords3D_rho = cp.sqrt(cp.sum(coords3D**2, axis = 1))
    
    coords3D_dec = cp.arccos(coords3D[:,2] / coords3D_rho )
    coords3D_ra = cp.sign(coords3D[:,1]) * cp.arccos( coords3D[:,0] / cp.sqrt( coords3D[:,0]**2 + coords3D[:,1]**2 ))
    
    coords3D_dec = 90. - (coords3D_dec * 180./cp.pi)
    coords3D_ra *= 180./cp.pi
    coords3D_ra += (coords3D_ra < 0)*360.
    
    return  cp.asnumpy(coords3D_ra), cp.asnumpy(coords3D_dec) # RA and DEC on the tangent plane



#%%
# C:/Users/champ/Astronomy/Reference/starmap_2020_64k.tif

def GenStarmapSlow(path_to_GaiaDR2_NASA_starmap = 'C:/Users/champ/Astronomy/Reference/starmap_2020_16k.tif',
               aug = 0.05):
    
    
    starmap_RGB = skio.imread(path_to_GaiaDR2_NASA_starmap).mean(axis = 2)
    
    orig_width = int(starmap_RGB.shape[1])
    orig_halfwidth = 0.5*orig_width
    aug_cols = int(aug * orig_width)
    
    
    starmap_RGB = np.fliplr(   np.roll(starmap_RGB, int(round(orig_halfwidth)) - 1 ))
    starmap_RGB = np.hstack([starmap_RGB[:,-aug_cols:], starmap_RGB, starmap_RGB[:,:aug_cols]])
    
    RA_coords = np.arange(0, orig_width, 1, dtype = float)
    RA_coords = 360.*((orig_halfwidth - RA_coords)/orig_width)
    RA_coords += 360.*(RA_coords < 0)
    RA_coords = np.roll(RA_coords, int(round(orig_halfwidth)) - 1 )[::-1]
    RA_coords = np.concatenate([RA_coords[-aug_cols:]-360., RA_coords, RA_coords[:aug_cols]+360])
    
    DEC_coords = 90. - 180. * np.arange(0, starmap_RGB.shape[0], 1, dtype = float)/starmap_RGB.shape[0]

    RA_coords_2D, DEC_coords_2D = np.meshgrid(RA_coords, DEC_coords)
    
    ## too slow!
    Interp_Starmap = sp.interpolate.RectBivariateSpline(
                                                        RA_coords, #np.arange(ImageCols.shape[1]), 
                                                        DEC_coords[::-1], #np.arange(ImageCols.shape[0]), 
                                                        starmap_RGB[::-1].T,
                                                        kx = 1,
                                                        ky = 1,
                                                        s = 0)
    
    ## faster but still too slow!
    # Interp_Starmap = sp.interpolate.NearestNDInterpolator(np.vstack((RA_coords_2D.ravel(), DEC_coords_2D.ravel())).T,
    #                                                     starmap_RGB.ravel())
    
    
    
    
    
    
    #Interp_Starmap = None
    ## usage of the interpolant:  Interp_Starmap.ev(RA_coords, DEC_coords) # output will have shape = RA_coords.shape

    
    return RA_coords, DEC_coords, RA_coords_2D, DEC_coords_2D, starmap_RGB, Interp_Starmap



def InitializeStarmap(path_to_EXR_GaiaDR2_NASA_starmap):
    
    mod_directory = os.path.split(fileio.__file__)[0]
    
    
    if 'starmap_grey.npy' in os.listdir(mod_directory):
        
        starmap_grey_fp = os.path.join(mod_directory, 'starmap_grey.npy')
        starmap_RGB_fp = os.path.join(mod_directory, 'starmap_RGB.npy')   
        
    else:
        
        print('Initializing starmap files (one time cost)...')
        starmap_RGB = cv2.imread(path_to_EXR_GaiaDR2_NASA_starmap,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
        starmap_grey = np.mean(starmap_RGB, axis = 2)
        
        starmap_RGB = np.array(starmap_RGB, dtype = np.float32)
        
        starmap_grey = np.array(starmap_grey, dtype = np.float32)
        
        starmap_grey_fp = os.path.join(mod_directory, 'starmap_grey.npy')
        starmap_RGB_fp = os.path.join(mod_directory, 'starmap_RGB.npy')   
        
        print('saving starmap files:', starmap_grey_fp, starmap_RGB_fp)
        np.save(starmap_grey_fp, starmap_grey)
        np.save(starmap_RGB_fp, starmap_RGB)
    
    
    return starmap_grey_fp, starmap_RGB_fp
    
    

def GenStarmapFast(path_to_EXR_GaiaDR2_NASA_starmap = 'C:/Users/champ/Astronomy/Reference/starmap_2020_16k.exr', mode = 'grey'):
    """Generates the interpolator for use in evaluating at image celestial 
    coordinates to retrieve the NASA Deep Star Map at the image location. 
    
    NOTE: upon the first time calling this function a large (nearly gigabyte) 
    serialized numpy array will be created that allows for very fast loading 
    on future function calls (this fuinction calls InitializeStarmap).  
    During the first call the exr file will be parsed
    and preppared for the creation fo an inteprolator. 

    Parameters
    ----------
        path_to_EXR_GaiaDR2_NASA_starmap : string, filepath
            Full file path to the exr file containing the NASA Deep Sky Map
            (obtain from https://svs.gsfc.nasa.gov/4851)
        mode : string
            set this to 'grey' to return a greytscale version of the starmap 
            instead of RGB.

    Returns
    -------
        EvalStarmap : function, EvalStarmap(eval_RA, eval_DEC)
            Provide RA and DEC coordinates to thsi function and it will evaluate 
            the NASA deep star map at the provided coordinates. 

    """    
    
    starmap_grey_fp, starmap_RGB_fp = InitializeStarmap(path_to_EXR_GaiaDR2_NASA_starmap)
    
    
    if mode == 'grey':
        
        starmap_image = np.load(starmap_grey_fp)
        
    else:
        
        starmap_image = np.load(starmap_RGB_fp)
        
    orig_width = int(starmap_image.shape[1])
    orig_halfwidth = 0.5*orig_width
    
    starmap_image = (  np.roll(starmap_image, int(round(orig_halfwidth)) - 1 , axis = 1)) 
    
    ## flip the RA axis so that it is in ascending order
    if mode == 'grey':
        starmap_image = starmap_image[:,::-1]
    else:
        starmap_image = starmap_image[:,::-1]
        
    # flip dec axis so that the data is on declination ascending order
    starmap_image = starmap_image[::-1]
    
    
    RA_coords = np.arange(0, orig_width, 1, dtype = float)
    RA_coords = 360.*((orig_halfwidth - RA_coords)/orig_width)
    RA_coords += 360.*(RA_coords < 0)
    RA_coords = np.roll(RA_coords, int(round(orig_halfwidth)) - 1 )[::-1]
    RA_coords[-1] = 99999
    
    DEC_coords = 90. - 180. * np.arange(0, starmap_image.shape[0], 1, dtype = float)/starmap_image.shape[0]

    # flip so that the data is on declination ascending order
    DEC_coords = DEC_coords[::-1]
    DEC_coords[-1] = 99999
    
    def EvalStarmap(eval_RA, eval_DEC):
        RA_insert = RA_coords.searchsorted(eval_RA)
        DEC_insert = DEC_coords.searchsorted(eval_DEC)
        #RA_insert[RA_insert >= starmap_RGB.shape[1]] = 0
        #DEC_insert[DEC_insert >= starmap_RGB.shape[0]] = 0
        if mode == 'grey':
            return starmap_image[DEC_insert, RA_insert]
        else:
            return starmap_image[DEC_insert, RA_insert,:]
        
   
    return EvalStarmap

#%%

def InitializeHAmap(path_to_Finkbeiner_allSky_Halpha, dec_res = 2**12, ra_res = 2**13):
    
    mod_directory = os.path.split(fileio.__file__)[0]
    
    res_str = str(dec_res) + 'x' + str(ra_res)
    
    if 'HAmap_RADEC_' + res_str + '.npy' in os.listdir(mod_directory):
        
        HAmap_fp = os.path.join(mod_directory, 'HAmap_RADEC_' + res_str + '.npy')
        RA_coords_fp = os.path.join(mod_directory, 'HAmap_RAcoords_' + res_str + '.npy')
        DEC_coords_fp = os.path.join(mod_directory, 'HAmap_DECcoords_' + res_str + '.npy') 
        
    else:
        
        print('Initializing HA RA/DEC files (one time cost, this will take a few minutes)...')


        ##############################################
        ### proc all-sky HA Map
        ## approach:  read->coords->interpolant->interpolate->save
        
        
        hdul = fits.open(path_to_Finkbeiner_allSky_Halpha)
        
        Ha_map = hdul[0].data
    
        map_Gal_Lat = np.linspace(-90,90, num = Ha_map.shape[0])
        map_GAL_Long = np.linspace(180,-180+.04167149, num = Ha_map.shape[1])
        map_GAL_Long[map_GAL_Long < 0 ] += 360
        
        SI = np.argsort(map_GAL_Long)
        map_GAL_Long = map_GAL_Long[SI]
        Ha_map = Ha_map[:,SI]
        
        Ha_map_padd = np.vstack([Ha_map[-200:-1],Ha_map,Ha_map[1:200]])
        Ha_map_padd = np.hstack([Ha_map_padd[:,-200:],Ha_map_padd,Ha_map_padd[:, :200]])
    
        map_GAL_Long = np.concatenate([map_GAL_Long[-200:]-360, map_GAL_Long, map_GAL_Long[:200]+360])
        map_Gal_Lat = np.concatenate([map_Gal_Lat[-200:-1]-180, map_Gal_Lat, map_Gal_Lat[1:200]+180])
        
        map_GAL_Long_2D, map_Gal_Lat_2D = np.meshgrid(map_GAL_Long, map_Gal_Lat)
        
        # construct the interpolant
        Interp_BiSS = sp.interpolate.RectBivariateSpline(map_GAL_Long, #np.arange(ImageCols.shape[1]), 
                                                           map_Gal_Lat, #np.arange(ImageCols.shape[0]), 
                                                           Ha_map_padd.T,
                                                           kx = 3,
                                                           ky = 3)
        
        ## Construct the @D grid of RA/DEC coords to evaluate the map upon
        RA_eval = np.linspace(0.0, 360., num = ra_res)
        DEC_eval = np.linspace(-90., 90., num = dec_res)
        
        RA_eval_2D, DEC_eval_2D = np.meshgrid(RA_eval, DEC_eval)
        
        ## Convert the RA/DEC coords to the Galactic coords used by the HAlpha map
        ICRS_all = SkyCoord(ra = RA_eval_2D.ravel(), dec = DEC_eval_2D.ravel(), frame = 'icrs', unit=u.deg)
    
        ICRS_all_gal_long = ICRS_all.galactic.l.deg
        ICRS_all_gal_lat = ICRS_all.galactic.b.deg
        
        ## Evaluate the HA interpolant at the ra/dec grid
        
        HAmap_RADEC = Interp_BiSS.ev(ICRS_all_gal_long, ICRS_all_gal_lat)
        
        HAmap_RADEC = HAmap_RADEC.reshape(RA_eval_2D.shape)
        
        HAcut_high = np.percentile(HAmap_RADEC, 99.99)
        
        HAmap_RADEC/=HAcut_high
        
        HAmap_RADEC = np.array(HAmap_RADEC, dtype = np.float32)

        #PlotImage(Interp_grid, colormap = plt.cm.inferno, max_color_val = 200)
        
        ################################################
        
        HAmap_fp = os.path.join(mod_directory, 'HAmap_RADEC_' + res_str + '.npy')
        RA_coords_fp = os.path.join(mod_directory, 'HAmap_RAcoords_' + res_str + '.npy')  
        DEC_coords_fp = os.path.join(mod_directory, 'HAmap_DECcoords_' + res_str + '.npy') 
        
        print('saving HA RA/DEC files:', HAmap_fp, RA_coords_fp, DEC_coords_fp)
        np.save(HAmap_fp, HAmap_RADEC)
        np.save(RA_coords_fp, RA_eval)
        np.save(DEC_coords_fp, DEC_eval)
  
    
    return HAmap_fp, RA_coords_fp, DEC_coords_fp
    

    
def GenHAmapFast(path_to_Finkbeiner_allSky_Halpha = 'C:/Users/champ/Astronomy/Reference/Halpha_map.fits', 
                 dec_res = 2**12, 
                 ra_res = 2**13):
    """Generates the interpolator for use in evaluating at image celestial 
    coordinates to retrieve the WHAM H-alphar Map at the image location. 
    
    NOTE: upon the first time calling this function a large (nearly gigabyte) 
    serialized numpy array will be created that allows for very fast loading 
    on future function calls (this function calls InitializeHAmap).  During 
    the first call the fits file will be parsed
    and preppared for the creation fo an inteprolator. 

    Parameters
    ----------
        path_to_Finkbeiner_allSky_Halpha : string, filepath
            Full file path to the fits file containing the H-alpha Sky Map
            (obtain from: https://faun.rc.fas.harvard.edu/dfink/skymaps/halpha/data/v1_1/index.html)
        dec_res : int
            declination resolution to prepare the interpolator at
        ra_res : int
            right ascention resolution to prepare the interpolator at (rec: 2x the dec res)
    Returns
    -------
        EvalHAmap : function, EvalHAmap(eval_RA, eval_DEC)
            Provide RA and DEC coordinates to this function and it will evaluate 
            the H-alpha map at the provided coordinates. 

    """       
    HAmap_fp, RA_coords_fp, DEC_coords_fp = InitializeHAmap(path_to_Finkbeiner_allSky_Halpha, 
                                                            dec_res = dec_res, 
                                                            ra_res = ra_res)
    
    HAmap_image = np.load(HAmap_fp)
    
    RA_coords = np.load(RA_coords_fp)
    DEC_coords = np.load(DEC_coords_fp)
    
    RA_coords[-1] = 99999
    DEC_coords[-1] = 99999
    
    def EvalHAmap(eval_RA, eval_DEC):
        RA_insert = RA_coords.searchsorted(eval_RA)
        DEC_insert = DEC_coords.searchsorted(eval_DEC)
        return HAmap_image[DEC_insert, RA_insert]
    
    return EvalHAmap

    

#%%



def SecondaryTestStarmap(starmap_image, camera_image_grey, threshold_perc = 95):
    
    if len(starmap_image.shape) > 2:
        starmap_grey = starmap_image.mean(axis = 2) 
        starmap_bool = (starmap_grey > np.percentile(starmap_grey, threshold_perc)) 
    else:
        starmap_bool = (starmap_image > np.percentile(starmap_image, threshold_perc))
        
    image_bool = 1*(camera_image_grey > np.percentile(camera_image_grey, threshold_perc)) #- 1
    
    LR0_UD0 = np.sum(starmap_bool * image_bool)
    LR1_UD0 = np.sum(starmap_bool[:,::-1] * image_bool)
    LR0_UD1 = np.sum(starmap_bool[::-1,:] * image_bool)
    LR1_UD1 = np.sum(starmap_bool[::-1,::-1] * image_bool)
    
    secondary_params = [(1,1), (1,-1), (-1,1), (-1,-1)]
    secondary_tests = np.array([LR0_UD0, LR1_UD0, LR0_UD1, LR1_UD1])
    
    best_score = secondary_tests.max()
    best_score_index = np.argmax(secondary_tests)
    best_params = secondary_params[best_score_index]
    print('Starmap secondary test results (% max score):',  np.round(100.*secondary_tests/best_score, 2))
    if best_score_index != 0:
        print('starmap flipped!  Use [::'+ str(best_params[0]) +',::'+ str(best_params[1]) +'] to repair')
    
    else:
        return starmap_image, best_params
    
    return starmap_image[::best_params[0],::best_params[1]], best_params


    
#%%


if __name__ == '__main__':
    
    
    T11 = time.time()
    
    
    test_files = ['C:/Users/champ/Astronomy/Data/RMSS_Lagoon_Trifid_Swan_Eagle/2023-06-16_22-44-30__-9.90_5.00s_0004_2.61_$$ECCENTRICITY$$_lagoon_swan_eagle_trifid.tif', #Lagoon
                  'C:/Users/champ/Astronomy/Data/Oph_20230525/2023-05-26_00-31-45__-9.60_5.00s_0008_2.95_$$ECCENTRICITY$$_Ophiuchus.fits',
                  'C:/Users/champ/Astronomy/Data/RMSS_NA_Nebula/2023-06-16_22-08-15__-10.00_5.00s_0256_2.50_$$ECCENTRICITY$$_NA_Nebula_RMSS.tif',
                  'F:/Astronomy/Data/HI2023_M42_g100/2023-12-09_06-53-31__-10.00_10.00s_0017_2.38_$$ECCENTRICITY$$_M42_HI.tif',
                  'F:/Astronomy/Data/RosettePlus_20231211_g100/2023-12-11_05-32-58__-9.80_10.00s_0010_2.66_$$ECCENTRICITY$$_RosettePlus.tif',
                  'C:/Users/champ/Astronomy/Data/HeartDoubleCluster3/2023-07-15_02-04-12__-6.50_5.00s_0496_2.33_$$ECCENTRICITY$$_Heart_DoubleCluster.tif']
    
    Labels = ['Lagoon', 'Rho Ophiuchi', 'North American Nebula', 'M42 Region', 'Rosette Cone', 'Heart, Soul, Double Cluster']
    
    image_index = 0
    
    image_label = Labels[image_index]
    image_fp = test_files[image_index]
    
    image_rgb, image_bayer = ReadASI_TIF_FITS(test_files[image_index])
    image_gray = image_rgb.mean(axis = 2)
    
    image_gray_norm = 2**16*(image_gray - np.percentile(image_gray, 5))
    image_gray_norm[image_gray_norm < 1] = 1
    
    image_gray_norm = np.log2(image_gray_norm)
    image_gray_norm /= 15
    
    T12 = time.time()
    
    ImageRows, ImageCols = GetImageCoords(AnalysisImage=image_rgb)
    
    T13 = time.time()
    
    platesolve_solution = PlateSolve(image_fp, 
                   ra_guess = None, dec_guess = None, 
                   fov = 8.85 , 
                   search = 170, 
                   astap_executable = '"C:/Program Files/astap/astap_cli.exe"')
    
    
    
        
    T14 = time.time()
  

    #deg_RA, deg_DEC = Pixels2RADEC_Transform3D(ImageRows.ravel(), ImageCols.ravel(), platesolve_solution, delta = 10)
    deg_RA, deg_DEC = Pixels2RADEC_Transform3D_cuda(ImageRows.ravel(), ImageCols.ravel(), platesolve_solution, delta = 10)
    
    T15 = time.time()
    #EvalStarmap = GenStarmapFast(mode = 'grey')
    EvalStarmap = GenStarmapFast(mode = 'rgb')
    
    T16 = time.time()
    starmap_at_image = EvalStarmap(deg_RA, deg_DEC)
    
    if len(starmap_at_image.shape)==2:
        starmap_at_image = starmap_at_image.reshape(image_rgb.shape)
    else:
        starmap_at_image = starmap_at_image.reshape(image_gray.shape)   
        
    T17 = time.time()
    
    
    starmap_at_image, secondary_test = SecondaryTestStarmap(starmap_at_image, 
                                                            image_gray, 
                                                            threshold_perc = 95)
    
    
    T18 = time.time()
    


    #PlotImage(image_gray_norm)
    #PlotImage(starmap_at_image)
    #PlotImage(starmap_at_image -image_gray )
    #PlotImage(np.log2(2**12*starmap_at_image.mean(axis = 2)))
    # PlotImageRGB(starmap_at_image*10)
    # PlotImageRGB(SaturationValueBoost(np.array(starmap_at_image, dtype = np.float32), 
    #                      saturation_boost = 1.0, 
    #                      value_boost = 10, 
    #                      boost_method = 'linear', 
    #                      return_HSV = False))
    

    # for mm in [1,2,4]:
    #     EvalHAmap = GenHAmapFast(path_to_Finkbeiner_allSky_Halpha = 'C:/Users/champ/Astronomy/Reference/Halpha_map.fits',
    #                              dec_res=2048*mm, ra_res=4096*mm)
        
    EvalHAmap = GenHAmapFast(path_to_Finkbeiner_allSky_Halpha = 'C:/Users/champ/Astronomy/Reference/Halpha_map.fits',
                             dec_res=8192, ra_res=16384)
    
    T19 = time.time()
    
    HAmap_at_image = EvalHAmap(deg_RA, deg_DEC)
    HAmap_at_image = HAmap_at_image.reshape(image_gray.shape)
    
    HAmap_at_image = HAmap_at_image[::secondary_test[0],::secondary_test[1]]
    HAmap_at_image[HAmap_at_image > 1.0] = 1.0
    
    T20 = time.time()
    
    
    
    # SynthView = deepcopy(starmap_at_image/.1)
    
    # SynthView[:,:,0] += HAmap_at_image
    # SynthView[:,:,2] += 0.33*HAmap_at_image
    
    #PlotImageRGB(SynthView)
    #PlotImage(HAmap_at_image, max_color_val = 1.1, colormap = plt.cm.gist_heat) # plt.cm.gist_heat, plt.cm.hot, plt.cm.inferno
    
    colored_HA = plt.cm.inferno(HAmap_at_image)[:,:,:3]

    mixed = 1.0 - (1.0 - colored_HA)*(1.0 - starmap_at_image)
    #PlotImageRGB(mixed)
    starmap_grey = starmap_at_image.mean(axis = 2)
    
    #PlotImage(starmap_grey, max_color_val = 0.1)
    
    
    p_star = np.percentile(starmap_grey, 50)
    
    p_HA = np.percentile(HAmap_at_image, 50)
    
    BG_DSO_STAR_bool = (starmap_grey < p_star) & (HAmap_at_image < 0.1 ) #p_HA)
    
    #PlotImage(1.0 - maximumDiskFilter(5, 1.0 - BG_DSO_STAR_bool))
    T21 = time.time()
    
    


    print('image reading and pre-processing:', T12-T11)
    print('Image RC coords:', T13-T12)
    print('plate solve:', T14-T13)
    print('RA DEC image coordinate calcs:', T15-T14)
    print('create starmap interpolant:', T16-T15)
    print('evaluate interpolant at image coordinates:', T17-T16)
    print('secondary testing of starmap:', T18-T17)
    print('prepare HA interpolator:', T19-T18)
    print('Eval HA map:', T20-T19)
    print('Combine starmap and HA map:', T21-T20)

    
    






























