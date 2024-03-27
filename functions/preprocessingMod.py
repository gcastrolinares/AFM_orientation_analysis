'''
@Djim de Ridder

This is the preprocessing module of septinNetworkAFM.
    This modules contains the variables:
        -
    This module contains the function:
        -FloatImgTo8Bit
'''
import pandas as pd
import numpy as np

from skimage.transform import resize

def FloatImgTo8Bit(im,
                   size,
                   config = None,
                   iConfig = 0,
                   invert = False):
    '''
    Convert float image into 8 bit

    Parameters
    ----------
    im : numpy.ndarray
        DESCRIPTION. image
    size : tuple
        DESCRIPTION. size 2 tuple with new image size (currenlty have only tried to downscale)
    config : pandas.core.frame.DataFrame
        DESCRIPTION. dataframe of config file used to scale image with fitted image height
        The default is None.
    iConfig : int
        DESCRIPTION. index of config
        The default is 0.
    invert : bool
        DESCRIPTION. should the iamge be inverted (we want white foreground)
        The default is False.

    Returns
    -------
    im : numpy.ndarray
        DESCRIPTION. image

    '''
    #TODO: if the config value is empty then it is a none imput
    if type(config) == pd.core.frame.DataFrame and not np.isnan(config['x4'][iConfig]):
        heightLimitUp = config['x4'][iConfig]+3*config['x5'][iConfig]
        im[im>heightLimitUp]=heightLimitUp
        heightLimitLow = config['x1'][iConfig]-3*config['x2'][iConfig]
        im[im<heightLimitLow]=heightLimitLow
        im = (im-heightLimitLow)/(heightLimitUp-heightLimitLow)
    if invert ==True:
        im = (im*(-1)+1) #invert simulated image
    
    if im.shape[0]/im.shape[1]==size[0]/size[1]:
        im = resize(im,size)
    else: #if the ratio of x and y does not match we need to crop the image
        if im.shape[1]/(im.shape[0]/size[0])<size[1]:
            im=im[0:int((im.shape[1]*size[0])/(size[1])),:]
            im = resize(im,size)
        elif im.shape[0]/(im.shape[1]/size[1])<size[0]: 
            im=im[:,0:int((im.shape[0]*size[1])/(size[0]))]
            im = resize(im,size)
        else:
            print('could not match ratio of shape between image and given dimensions')
    im = np.round((im-np.min(im))/(np.max(im)-np.min(im))*255)
    
    return im

        

