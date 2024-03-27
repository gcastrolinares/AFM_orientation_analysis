'''
@Djim de Ridder

This is the plotting module of septinNetworkAFM.
    This modules contains the variables:
        -
    This module contains the function:
        -LoadConfigFile
        -LoadAFMTxtFile
        -LoadSimPngFile
'''
import os
import re

import pandas as pd
import numpy as np

import skimage.io

def LoadConfigFile(fileFolder = os.path.join(os.getcwd(), 'input'),
                   nameConfig = "fitHeightDistributionshdn.xlsx"
                   ):
    '''
    Loads the excel config file here AFM image data is stored including the name of the images.

    Parameters
    ----------
    fileFolder : path, optional
        DESCRIPTION. Folder path where the code will look for the config file.
        The default is os.path.join(os.getcwd(), 'input').
    nameConfig : str, optional
        DESCRIPTION. Name of the excel config file
        The default is "fitHeightDistributionshdn.xlsx".

    Returns
    -------
    config : pandas.core.frame.DataFrame
        DESCRIPTION. dataframe of config file

    '''
    #load config file
    absolute_path_config = os.path.join(fileFolder,nameConfig)
    config = pd.read_excel(absolute_path_config,skiprows=1)
    return config


def LoadTxtFileFromConfig(config,
                   fileFolder = os.path.join(os.getcwd(), 'input'),
                   iConfig = 15
                   ):
    '''
    Loads the AFM coorelated with the "iConfig" index in the config dataframe.

    Parameters
    ----------
    config : pandas.core.frame.DataFrame
        DESCRIPTION. dataframe of config file
    fileFolder : path, optional
        Folder path where code will look for the image
        DESCRIPTION. The default is os.path.join(os.getcwd(), 'input').
    iConfig : int, optional
        DESCRIPTION. Index of image inside config file
        The default is 15.

    Returns
    -------
    im : numpy.ndarray
        DESCRIPTION. numpy array of loaded image

    '''
    #read config file
    nameI = config['name'][iConfig]
    ymin,ymax,xmin,xmax=[int(s) for s in re.findall(r'\b\d+\b',config['crop'][iConfig])]
    
    #load image from information config
    absolute_path_I = os.path.join(fileFolder,nameI)
    im =(np.loadtxt(absolute_path_I)*10**9)[ymin:ymax,xmin:xmax]
    
    return im

def LoadPngFile(nameI="simulatedSeptinNetwork.png",
                fileFolder = os.path.join(os.getcwd(), 'input'),
                ):
    '''
    This function loads and png image
    
    Parameters
    -------
    nameI : STR
        DESCRIPTION. name of png file
    fileFolder: path
        DESCRIPTION. path of folder where image is stored
        The default is os.path.join(os.getcwd(), 'input').
    
    Returns
    -------
    im: numpy.ndarray
        DESCRIPTION numpy array of image (most likely 3d since images have a rgb value)
    
    '''
    #load simulated image
    absolute_path_I = os.path.join(fileFolder,nameI)
    im = np.array(skimage.io.imread(absolute_path_I,as_gray=True),dtype=np.uint8)
    return im