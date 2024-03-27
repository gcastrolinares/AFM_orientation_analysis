'''
@Djim de Ridder

This is the analysing module of septinNetworkAFM.
    This modules contains the variables:
        -
    This module contains the function:
        -CalculateStructureTensor
        -CalculateOrientations
        -FitHeightProfile
'''
import numpy as np
from skimage.draw import disk
import orientationpy

from ast import literal_eval
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def CalculateStructureTensor(im,
                             sig =1,
                             mode = "gaussian"
                             ):
    '''
    Calculate the structure tensor of an image using the mode given

    Parameters
    ----------
    im : numpy.ndarray
        DESCRIPTION. image
    sig : float (or int)
        DESCRIPTION. standard deviation necessary to calculate the image gradient
        The default is 1
    mode : str
        DESCRIPTION. choose method to estimate image gradient choose between "finite_difference", "splines", "gaussian" or "test"
        The default is "gaussian".

    Returns
    -------
    structureTensor : TYPE
        DESCRIPTION.

    '''
    if mode=="test":
        fig = plt.figure(tight_layout=True)
        fig.set_size_inches(15,10) #height, width
        gs6 = gridspec.GridSpec(3,2)
        for n, modes in enumerate(["finite_difference", "splines", "gaussian"]):
            #estimation gradient
            Gy, Gx = orientationpy.computeGradient(im, mode=modes)
            
            ax = fig.add_subplot(gs6[n,0])
            ax.set_title(f"{mode}-Gy")
            ax.imshow(Gy, cmap="coolwarm", vmin=-64, vmax=64)
    
            ax = fig.add_subplot(gs6[n,1])
            ax.set_title(f"{mode}-Gx")
            ax.imshow(Gx, cmap="coolwarm", vmin=-64, vmax=64)
            
            #structure tensor
            structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=sig)
    else:
        Gy, Gx = orientationpy.computeGradient(im, mode=mode)
        structureTensor = orientationpy.computeStructureTensor([Gy, Gx], sigma=sig)
    return structureTensor


def CalculateOrientations(structureTensor,
                          mask = True
                          ):
    '''
    Calculates orientations for structure tensor

    Parameters
    ----------
    structureTensor : numpy.ndarray
        DESCRIPTION. 3xNxM array for each pixels strucutre tensor (Ixx,Iyy,Ixy)
    mask : bool
        DESCRIPTION. boolean if you cuts out a circle in the image
        The default is True.

    Returns
    -------
    orientations : dict
        DESCRIPTION. orientations dict with numpy.ndarray for the ['theta'],['coherency'],['energy'] 

    '''
    orientations = orientationpy.computeOrientation(structureTensor, computeEnergy=True, computeCoherency=True)
    # if mask:
    #     image_height, image_width = structureTensor.shape[1:3]
    #     radius_percentage = 0.2  # Adjust this value as needed
    #     # Calculate radius based on image size
    #     radius = min(image_height, image_width) * radius_percentage
    #     center_x, center_y = image_width // 2, image_height // 2
    # if mask == True:
    #     minAx = min(structureTensor.shape[1:3])
    #     center = (structureTensor.shape[1] // 2, structureTensor.shape[2] // 2)
    #     radius = (minAx // 2) - 2
    #
    #     # Check if the mask radius exceeds image dimensions
    #     if radius < minAx // 2:
    #         rr, cc = disk(center, radius, shape=structureTensor.shape[1:3])
    #
    #         mask = np.zeros(structureTensor.shape[1:3], dtype=np.uint8)
    #         mask[rr, cc] = 1
    #         mask = mask > 0
    #
    #         # Apply the mask and remove NaN values
    #         orientations['theta'][mask] = np.nan
    #         orientations['coherency'][mask] = np.nan
    #         orientations['energy'][mask] = np.nan
    #         # orientations['theta'] = orientations['theta'][~np.isnan(orientations['theta'])]
    #         # orientations['coherency'] = orientations['coherency'][~np.isnan(orientations['coherency'])]
    #         # orientations['energy'] = orientations['energy'][~np.isnan(orientations['energy'])]
    #     else:
    #         print("Mask radius exceeds image dimensions. Cannot apply mask.")
    #
    # return orientations

#Dijms code
    if mask == True:
        # ---to avoid edge artifact make circular mask
        minAx = np.min(structureTensor.shape[1:3])
        rr, cc = disk((structureTensor.shape[1] / 2, structureTensor.shape[2] / 2), (minAx / 2) - 2,
                      shape=structureTensor.shape[1:3])

        mask = np.zeros(structureTensor.shape[1:3], dtype=np.uint8)
        mask[rr, cc] = 1
        mask = mask > 0
        # print(rr, cc.shape)

        orientations['theta'][~mask] = np.nan
        orientations['coherency'][~mask] = np.nan
        orientations['energy'][~mask] = np.nan


    return orientations
def FitHeightProfile(im,
                     config,
                     iConfig=15
                     ):
    '''
    Fit the sum of two gaussian over a histogram of an image

    Parameters
    ----------
    im : numpy.ndarray
        DESCRIPTION. image
    config : pandas.core.frame.DataFrame
        DESCRIPTION. dataframe of config file
    iConfig : int
        DESCRIPTION. index of image to load estimation information
        The default is 15.

    Raises
    ------
    ValueError
        DESCRIPTION. 'Inside config file give an estimation of fitted values (muBG,sigmaBG,intenistyBG,muNetwork,sigmaNetwork,intenistyNetwork)'

    Returns
    -------
    params : tuple
        DESCRIPTION. size 6 parameter values of two gaussians (muBG,sigmaBG,intenistyBG,muNetwork,sigmaNetwork,intenistyNetwork)'

    '''
    expected = literal_eval(config['params_estimation'][iConfig])
    if isinstance(expected,tuple) and len(expected)==6:
        y,x = np.histogram(im.ravel(),bins=100,density=True)
        x=(x[1:]+x[:-1])/2 # correct hist data
        
        def gauss(x,mu,sigma,A):
            return A*np.exp(-(x-mu)**2/2/sigma**2)

        def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
            return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
        params,cov=curve_fit(bimodal,x,y,expected)
    else:
        raise ValueError('Inside config file give an estimation of fitted values (muBG,sigmaBG,intenistyBG,muNetwork,sigmaNetwork,intenistyNetwork')
    
    return params