# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:11:17 2024

@author: champ
"""


import cv2
import numpy as np
from copy import deepcopy







def RGB2Luminance(R_data, G_data=[], B_data=[]):
    
    if len(R_data.shape) == 3:
        L_data = 0.2126 * R_data[:,:,0] + 0.7152*R_data[:,:,1] + 0.0722*R_data[:,:,2]
    else:
        L_data = 0.2126 * R_data + 0.7152*G_data + 0.0722*B_data
    
    return L_data

def RGB2Intensity(R_data, G_data=[], B_data=[]):
    if len(R_data.shape) == 3:
        I_data = (1.0/3.0)*(R_data.sum(axis = 2)) 
    else:
        I_data = (R_data + G_data + B_data)/3.0 
    
    return I_data


    
    
def RGB2HSV(imRGB):
    
    return cv2.cvtColor(np.array(imRGB, dtype = np.float32), cv2.COLOR_RGB2HSV)

def HSV2RGB(imHSV):
    
    return cv2.cvtColor(np.array(imHSV, dtype = np.float32), cv2.COLOR_HSV2RGB)


    
# def RGB2HSI(imRGB):
    
#     return cv2.cvtColor(np.array(imRGB, dtype = np.float32), cv2.COLOR_RGB2HSV)

# def HSI2RGB(imHSV):
    
#     return cv2.cvtColor(np.array(imHSV, dtype = np.float32), cv2.COLOR_HSV2RGB)



    
def RGB2HLS(imRGB):
    
    return cv2.cvtColor(np.array(imRGB, dtype = np.float32), cv2.COLOR_RGB2HLS)

def HLS2RGB(imHLS):
    
    return cv2.cvtColor(np.array(imHLS, dtype = np.float32), cv2.COLOR_HLS2RGB)








def ApplyColorTrans(conv_mat, RGB_Nx3):
   
    return np.dot(conv_mat, RGB_Nx3.T).T



def ColorConvolveImage(conv_mat, imageRGB):
   
    frame_shape = imageRGB[:,:,0].shape
    ID = deepcopy(np.vstack([imageRGB[:,:,0].ravel(), imageRGB[:,:,1].ravel(), imageRGB[:,:,2].ravel()])).T
   
   
    cID = ApplyColorTrans(conv_mat, ID)
   
    conv_image_RGB = np.stack([cID[:,0].reshape(frame_shape),
                               cID[:,1].reshape(frame_shape),
                               cID[:,2].reshape(frame_shape)], axis = 2)
    
    
    conv_image_RGB[conv_image_RGB < 0] = 0
   
    return conv_image_RGB
   
    
### full spectrum color convolution matrix computed by finding the optimal linear
### trasnformation that maps ray camera RGB to a known answer color chart.
### This is specific to the ASI2600MCPro RGB camera
full_spec_ASI2600MC_conv_mat =    np.array([[ 1.        , -0.1885989 ,  0.33030315],
                                            [ 0.06851153,  0.57304046,  0.14495132],
                                            [ 0.27286604, -0.32110659,  1.53035189]]) 

### L-Enhance color convolution matrix computed by finding the optimal linear
### trasnformation that maps ray camera RGB to a known answer color chart. 
### This is specific to the ASI2600MCPro RGB camera combined with an L-Enhance filter. 
LEnhance_ASI2600MC_conv_mat = np.array([[ 1.        ,  1.23200262, -1.45530737],
                                        [-0.07611606,  4.71648981, -5.49477104],
                                        [ 0.32520944, -1.04157622,  1.97164413]])







def RGB2HSI(imageRGB):
    
    this_sensor_shape_rgb = imageRGB.shape
    
    Intensity = (1.0/3.0)*(imageRGB[:,:,0] + imageRGB[:,:,1] + imageRGB[:,:,2])
    
    BdomG_Bool = imageRGB[:,:,1] < imageRGB[:,:,2]
    
    Hue = imageRGB[:,:,0]**2 + imageRGB[:,:,1]**2 + imageRGB[:,:,2]**2
    Hue -= (imageRGB[:,:,0]*imageRGB[:,:,1] + imageRGB[:,:,0]*imageRGB[:,:,2] + imageRGB[:,:,1]*imageRGB[:,:,2])
    Hue = np.sqrt(Hue)

    print('zero hue:', np.sum(Hue == 0), np.sum(np.isnan(Hue)))
    
    Hue = (imageRGB[:,:,0] - 0.5*imageRGB[:,:,1] - 0.5*imageRGB[:,:,2]) / Hue
    
    ## fix the floting point errors
    Hue[Hue < -1.0] = -1.0
    Hue[Hue > 1.0] = 1.0
    
    Hue = 180./np.pi * np.arccos(Hue)

    Hue[BdomG_Bool] = 360. - Hue[BdomG_Bool]

    Hue[Hue >= 360.] = Hue[Hue >= 360.] - 360.
    ## need to fix this to that if Intensity == 0 we don't get NaN's
    Saturation = 1.0 - np.min(imageRGB, axis = 2) / Intensity
    
    HSI = np.zeros(this_sensor_shape_rgb, dtype = float)
    
    HSI[:,:,0] = Hue
    HSI[:,:,1] = Saturation
    HSI[:,:,2] = Intensity
    
    return HSI



def HSI2RGB(imageHSI):
    this_sensor_shape_rgb = imageHSI.shape
    RGB = np.zeros(this_sensor_shape_rgb, dtype = float)
    
    _H = imageHSI[:,:,0]
    _Hrad = (np.pi/180) * _H
    _S = imageHSI[:,:,1]
    _I = imageHSI[:,:,2]
    
    # Case 1: Hue == 0
    h_bool = _H == 0.0
    RGB[:,:,0][h_bool] = (_I + 2.0 * _I * _S)[h_bool]
    RGB[:,:,1][h_bool] = (_I + _I * _S)[h_bool]
    RGB[:,:,2][h_bool] = (_I + _I * _S)[h_bool]
    
    # Case 2: 0 < Hue < 120
    h_bool = (0 < _H) & (_H < 120)
    RGB[:,:,0][h_bool] = (_I + _I * _S * (np.cos(_Hrad)/np.cos(60.*np.pi/180. - _Hrad)))[h_bool]
    RGB[:,:,1][h_bool] = (_I + _I * _S * (1. - np.cos(_Hrad)/np.cos(60.*np.pi/180. - _Hrad)))[h_bool]
    RGB[:,:,2][h_bool] = (_I - _I * _S    )[h_bool]
    
    # Case 3: Hue == 120
    h_bool = _H == 120.
    RGB[:,:,0][h_bool] = (_I - _I * _S)[h_bool]
    RGB[:,:,1][h_bool] = (_I + 2.0 * _I * _S)[h_bool]
    RGB[:,:,2][h_bool] = (_I - _I * _S   ) [h_bool]
    
    # Case 4: 120 < Hue < 240
    h_bool = (120. < _H) & (_H < 240.)
    RGB[:,:,0][h_bool] = (_I - _I * _S  )[h_bool]
    RGB[:,:,1][h_bool] = (_I + _I * _S * (np.cos(_Hrad - 120.*np.pi/180.)/np.cos(np.pi - _Hrad)))[h_bool]
    RGB[:,:,2][h_bool] = (_I + _I * _S * (1. - np.cos(_Hrad - 120.*np.pi/180.)/np.cos(np.pi - _Hrad))   )[h_bool]
    
    # Case 5: Hue == 240
    h_bool = _H == 240.
    RGB[:,:,0][h_bool] = (_I - _I * _S)[h_bool]
    RGB[:,:,1][h_bool] = (_I - _I * _S)[h_bool]
    RGB[:,:,2][h_bool] = (_I + 2.0 * _I * _S   )    [h_bool]
    
    # Case 4: 240 < Hue < 360
    h_bool = (240. < _H) & (_H < 360.)
    RGB[:,:,0][h_bool] =  (_I + _I * _S * (1. - np.cos(_Hrad - 240.*np.pi/180.)/np.cos(300.*np.pi/180. - _Hrad)) )[h_bool]
    RGB[:,:,1][h_bool] =  (_I - _I * _S )[h_bool]
    RGB[:,:,2][h_bool] =  (_I + _I * _S * (np.cos(_Hrad - 240.*np.pi/180.)/np.cos(300.*np.pi/180. - _Hrad))  )[h_bool]
    
    RGB[RGB < 0.] = 0.0
    RGB[RGB > 1.0] = 1.0
    
    return RGB


























