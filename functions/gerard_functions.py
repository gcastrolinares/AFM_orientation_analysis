'''
@Gerard Castro-Linares

'''

import os
import numpy as np
import functions as f
import sys
from skimage.transform import resize
from skimage.draw import disk
from scipy.optimize import curve_fit
import re
from PIL import Image

def load_AFM_txt(n):
    '''
    Loads a AFM image from a txt file.
    Parameters
    ----------
    n : int
        DESCRIPTION. Index number of the image to open in the config file.

    Returns
    -------
    imAFM: numpy.ndarray
        DESCRIPTION. Region of the original image determined by the pixels in dim.
    dim: dict
        DESCRIPTION. dimensions of the image.
    px_size: float
        DESCRIPTION. Pixel size of the image in micrometers per pixel.
    config['name'][indexConfig]: str
        DESCRIPTION. file name.
    width: float
        DESCRIPTION. Width of the original image
    '''
    global indexConfig
    indexConfig = n
    # Set the folder where the txt file is
    folderAFM = os.path.join(os.path.normpath(sys.path[3] + "/" + "/input"))
    # load config file
    global config
    config = f.loadingMod.LoadConfigFile()
    # get the crop dimensions saved in the config file and stored them in a dict
    ymin, ymax, xmin, xmax = [int(s) for s in re.findall(r'\b\d+\b', config['crop'][indexConfig])]
    dim = {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax}
    # get the width of the image stored in the txt file of the image
    with open(folderAFM + "//" + config['name'][indexConfig]) as fp:
        for i, line in enumerate(fp):
            if i == 1:
                width = float(line.split(" ")[2])
                break
    # get the original image size in px stored in the file name
    pattern = re.compile(r'(\d+)px')
    match = pattern.search(config['name'][indexConfig])
    px_value = int(match.group(1))
    # calculate the pixel size
    px_size = width / px_value
    # load AFM image
    imAFM = f.loadingMod.LoadTxtFileFromConfig(config, fileFolder=folderAFM, iConfig=indexConfig)
    return imAFM, dim, px_size, config['name'][indexConfig], width


def pre_orientation_AFM(n):
    '''
    Convert float image into 8 bit.
    For the orientation information we need a 8 bit downscaled figure where low intenisty correlates with the background. This step might not be necessary however it removes computational time and allows the use of any 8 bit image tool.

    Parameters
    ----------
    n : int
        DESCRIPTION. Index number of the image to open in the config file.

    Returns
    -------
    imAFMp: numpy.ndarray
        DESCRIPTION. Crop of the original image in 8 bit.
    px_size: float
        DESCRIPTION. Pixel size of the image in micrometers per pixel.
    name: str
        DESCRIPTION. file name.
    width: float
        DESCRIPTION. Width of the original image
    '''
    # load crop of the AFM image
    imAFM, dim, px_size, name, width = load_AFM_txt(n)
    # gets the dimensions of the image
    S = [dim['xmax'] - dim['xmin'], dim['ymax'] - dim['ymin']]
    # check whether image has NaN values
    has_nans = np.isnan(imAFM).any()
    if has_nans:
        # if the image has NaN values, skip it.
        print("Image data contains NaN values")
    else:
        # if not, transform it to 8 bit.
        imAFMp = f.preprocessingMod.FloatImgTo8Bit(im=imAFM,
                                                   size=(S[1], S[0]),
                                                   config=config,
                                                   iConfig=n
                                                   )
    return imAFMp, px_size, name, width


def orientation_AFM(n):
    '''
    Calculates orientation image.

    Parameters
    ----------
    n : int
        DESCRIPTION. Index number of the image to open in the config file.

    Returns
    -------
    orientationsAFM: dict
        DESCRIPTION.  orientations dict with numpy.ndarray for the ['theta'],['coherency'],['energy']
    px_size: float
        DESCRIPTION. Pixel size of the image in micrometers per pixel.
    name: str
        DESCRIPTION. file name.
    width: float
        DESCRIPTION. Width of the original image
    imAFMp: numpy.ndarray
        DESCRIPTION. Crop of the original image in 8 bit.
    '''
    # open 8bit AFm image
    imAFMp, px_size, name, width = pre_orientation_AFM(n)
    # compute structure tensor
    imAFMT = f.analysingMod.CalculateStructureTensor(imAFMp,
                                                     mode="gaussian",
                                                     )
    # compute orientations
    orientationsAFM = f.analysingMod.CalculateOrientations(imAFMT,
                                                           mask=False
                                                           )
    # save in a text file each of the images produced by the analysis: orientation, energy, and coherency.
    save_folder = os.path.join(os.path.normpath(sys.path[3]  + "/output"))
    for i in orientationsAFM.keys():
        name_file = str(n) + "_" + i + ".txt"
        np.savetxt(save_folder + "\\" + name_file, orientationsAFM[i], delimiter="\t")

    print("==== Complete the orientation for the AFM txt file! ====")

    return orientationsAFM, px_size, name, width, imAFMp


def pre_orientation_SIM(folder_dir, name):
    '''
    loads simulations images and convert them into 8 bit.

    Parameters
    ----------
    folder_dir: path
        DESCRIPTION. path of folder where image is stored.
    name: str
        DESCRIPTION. name of png file

    Returns
    -------
    imSIMp: numpy.ndarray
        DESCRIPTION. image in 8 bit.
    '''
    # load simulation image
    imSIM = f.loadingMod.LoadPngFile(name, folder_dir)
    # transform the background to be "dark"
    imSIM = 255 - imSIM
    # check whether image has nans
    has_nans = np.isnan(imSIM).any()
    if has_nans:
        # if it does, skip image
        print("Image data contains NaN values")
    else:  # If it does not
        # resize image to speed up the analysis
        imSIM_resized = resize(imSIM, (768, 768))
        # transform it to 8 bit.
        imSIMp = f.preprocessingMod.FloatImgTo8Bit(im=imSIM_resized,
                                                   size=(imSIM_resized.shape[1], imSIM_resized.shape[0]),
                                                   # size = (768, 768)
                                                   )
        # save resized 8 bit image
        im_save = Image.fromarray(imSIMp).convert('RGB')
        absolute_path_I = os.path.join(folder_dir, name[:-4] + "_resized_invert.png")
        im_save.save(absolute_path_I)
    return imSIMp


def orientation_SIM(name, type, run):
    '''
    Calculates orientation image.

    Parameters
    ----------
    name: str
        DESCRIPTION. name of png file
    type: str
        DESCRIPTION: type of simulation --> crossing ot alignment.
    run: str
        DESCRIPTION: run number as "run"+"number"
    Returns
    -------
    orientationsSIM: dict
        DESCRIPTION.  orientations dict with numpy.ndarray for the ['theta'],['coherency'],['energy']
    px_size: float
        DESCRIPTION. Pixel size of the image in micrometers per pixel.
    name: str
        DESCRIPTION. file name.
    width: float
        DESCRIPTION. Width of the original image
    imSIMp: numpy.ndarray
        DESCRIPTION. Original image in 8 bit.
    '''
    # Put together the input parameters to determine the file name.
    folder_dir = os.path.join(os.path.normpath(sys.path[3] + "/" + "/input" + "/" + type + "/" + run))
    # open image
    imSIMp = pre_orientation_SIM(folder_dir, name)
    # set width or the image, which is always 5.
    width = 5
    # calculate pixel size
    px_size = width / imSIMp.shape[1]
    # compute structure tensor
    imSIMt = f.analysingMod.CalculateStructureTensor(imSIMp,
                                                     mode="gaussian",
                                                     )
    # compute orientations
    orientationsSIM = f.analysingMod.CalculateOrientations(imSIMt,
                                                           mask=False
                                                           )

    # save in a text file each of the images produced by the analysis: orientation, energy, and coherency.
    save_folder = os.path.join(os.path.normpath(sys.path[3] + "/output"))

    for i in orientationsSIM.keys():
        np.savetxt(
            save_folder + "\\" + type + "_" + run + "_" + name[:-4] + "_" + str(i) + ".txt",
            orientationsSIM[i], delimiter="\t")

    print("==== Complete the orientation for the AFM txt file! ====")

    return orientationsSIM, px_size, name, width, imSIMp


def dot_unit(x, y):
    '''
    Returns the dot product of two angles or an angle and an array of angles in radians (considering they are unit vectors).
    Parameters
    ----------
    x: float
        DESCRIPTION. value of the first angle
    y: float or numpy.ndarray
        DESCRIPTION: if float, it is the second angle; and if numpy.ndarray, it is the array of angles
    Returns
    -------
    (np.cos(x)*np.cos(y))+(np.sin(x)*np.sin(y)): float or numpy.ndarray
            DESCRIPTION.  Result of the dot product
    '''
    return ((np.cos(x) * np.cos(y)) + (np.sin(x) * np.sin(y)))


def ring_mask(array, center, diameter, ring_size=3):
    '''
    Creates a ring-shaped mask of the image "array", centered in the pixe "center", with an inner diameter of "diameter", and a thickness of "ring_size",
    Parameters
    ----------
    array: numpy.ndarray
        DESCRIPTION. image to which the ring mask will be applied
    center: numpy.ndarray, tuple, or list
        DESCRIPTION. central pixel of the ring
    diamater: int
        DESCRIPTION. Inner diameter of the ring
    ring_size: int
        DESCRIPTION. Thickness of the ring.
        The default is 3
    Returns
    -------
    array: numpy.ndarray
        DESCRIPTION. original image (array) masked with the created ring.
    '''
    # Calculate the total diameter of the ring
    new_size = diameter + ring_size
    # create circular mask with a radius of the inner diameter of the ring over 2
    rr, cc = disk((center[0], center[1]), (diameter / 2), shape=array.shape)
    # create a circular mask with a radius of the total diameter over 2
    rr2, cc2 = disk((center[0], center[1]), (new_size / 2), shape=array.shape)
    # create a new empty image (all 0) with the dimensions of the original image
    mask = np.zeros(shape=array.shape, dtype=np.uint8)
    # set to a value of 1 the area inside the second (bigger) circular mask.
    mask[rr2, cc2] = 1
    # set back to a value of 0 the area inside the first (smaller) circular mask
    mask[rr, cc] = 0
    # transform the mask into a bool
    mask = mask > 0
    # apply mask on the image
    array[~mask] = np.nan
    return array


def select_random_pixels(image, diameter, number=5000):
    '''
    Selects "number" ammount of random pixels from "image". Select pixels that are further than "diameter" pixels away from the edge of the image.
    Parameters
    ----------
    image: numpy.ndarray
        DESCRIPTION. image.
    diamater: int
        DESCRIPTION. Distance from the edge within which no pixels will be selected.
    number: int
        DESCRIPTION. number of random pixels to select
        The default is 5000
    Returns
    -------
    array: numpy.ndarray
        DESCRIPTION. array containing the indices (x and y location) of each of the pixels.
    '''
    # Create a grid of indices
    rows, cols = np.indices(image.shape)

    # Compute the distance from each pixel to the nearest edge
    distances_to_edges = np.stack([rows, cols, image.shape[0] - rows - 1, image.shape[1] - cols - 1])

    # Compute the minimum distance along the third axis
    distances_to_nearest_edge = np.min(distances_to_edges, axis=0)

    # Create a binary mask based on the distance from the circular region around the edges
    mask = distances_to_nearest_edge < diameter

    # Apply the mask to the original image
    result = np.where(mask, 0, image)

    # calculate how many random pixels are selected.
    valid_pixels = np.argwhere(result != 0)
    # create empty array where the pixels will be stored.
    random_pixels = np.empty([0, 0])
    if valid_pixels.shape[0] <= 0:
        # if there are no pixels selected, give errors
        print('No pixels available!')
    elif valid_pixels.shape[0] < number:
        # if there are pixels, but less than the ammount asked, save them, but wive a warning error.
        print("less than", number, 'pixels avaliable, using', valid_pixels.shape[0], "pixels")
        number = valid_pixels.shape[0]
        random_indices = np.random.choice(valid_pixels.shape[0], size=number, replace=False)
        random_pixels = valid_pixels[random_indices]
    else:
        # if enough pixels have been selected, save them
        random_indices = np.random.choice(valid_pixels.shape[0], size=number, replace=False)
        random_pixels = valid_pixels[random_indices]
    return random_pixels


def FitFunction(x, C, L, C2):
    '''
    Fit function for the orientational decay
    Parameters
    ----------
    x: float
        DESCRIPTION. Distance from the central pixel.
    C: float
        DESCRIPTION. Initial value.
    L: float
        DESCRIPTION. Characteristic patch size, or lambda
    C2: float
        DESCRIPTION. Saturation value at which the decay stops.
    Returns
    -------
    (C2+((C-C2)*(np.exp(-x/L)))): float
        DESCRIPTION. Output value
    '''
    return (C2 + ((C - C2) * (np.exp(-x / L))))


def fit_AFM(x, y):
    '''
    Fitting of the orientational decay to the fit function
    Parameters
    ----------
    x: numpy.ndarray
        DESCRIPTION. Array containing the distances corresponding to each value in y.
    y: numpy.ndarray
        DESCRIPTION. Values of the orientational decay.
    Returns
    -------
    C: float
        DESCRIPTION. Initial value.
    L: float
        DESCRIPTION. Characteristic patch size, or lambda
    C2: float
        DESCRIPTION. Saturation value at which the decay stops.
    rmse_fit: float
        DESCRIPTION. Root-mean-square deviation between the observed orientational decay and the fitted data

    '''
    # guess initial values used for the fit
    p0 = [0.1, 5, 0.4]
    # fit the data to the function
    C, L, C2 = curve_fit(FitFunction, x, y, p0)[0]
    # calculate Root-mean-square deviation
    rmse_fit = np.sqrt(np.mean((y - FitFunction(x, C, L, C2)) ** 2))
    return C, L, C2, rmse_fit