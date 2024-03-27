'''
@Djim de Ridder

This is the plotting module of septinNetworkAFM.
    This modules contains the variables:
        -
    This module contains the function:
        -SettingUpPlot
        -PlotAfmImage
        -PlotGrayImage
        -PlotImageHistogram
        -PlotBimodalFit
        -PlotAfmProfilePlot
        -PlotHSBCircular_hist
        -Circular_hist @author: jwalton
        -PlotOrientationHistogram

'''
import numpy as np

from skimage.measure import profile_line

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import matplotlib.gridspec as gridspec
from matplotlib_scalebar.scalebar import ScaleBar

def SettingUpPlot(figGridx,
                  figGridy,
                  figSize=None):
    '''
    Setting up a figure with a coordinate grid

    Parameters
    ----------
    figGridx : int
        DESCRIPTION. size of figure grid in x
    figGridy : int
        DESCRIPTION. size of figure grid in y
    figSize : tuple
        DESCRIPTION. Size 2 tuple contaning figure size (width,height)

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. output figure
    gs : matplotlib.gridspec.GridSpec
        DESCRIPTION. output figure grid

    '''
    if figSize==None:
        figSize=(figGridx*5,figGridy*5)
        
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(figSize) #height, width
    gs = gridspec.GridSpec(figGridy,figGridx)
    return fig,gs

def PlotAfmImage(fig,
                 gs,
                 AFMimage,
                 clow=0,
                 chigh=30,
                 widthpx=512,
                 widthnm=10000
                 ):
    '''
    Plot AFM image with height-color bar
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. subplot inside figure grid (gridspec.GridSpec[0,0])
    AFMimage : numpy.ndarray
        DESCRIPTION. image
    clow : float
        DESCRIPTION. lower height limit
        The default is 0.
    chigh : float
        DESCRIPTION. heigher height limit
        The default is 30.
    widthpx : float
        DESCRIPTION. width of known distance in pixels
        The default is 512.
    widthnm : float
        DESCRIPTION. width of known distance in nm
        The default is 10000.

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. Figure
    ax : matplotlib.axes._axes.Axes
        DESCRIPTION. Axes of figure

    '''
    ax = fig.add_subplot(gs)
    
    cax=ax.imshow(AFMimage,'afmhot')
    cax.set_clim(clow, chigh)
    ax.axis('off')
    
    scalebar = ScaleBar(widthnm/widthpx, "nm",length_fraction=0.3,width_fraction=1/30,color='w',frameon=False,location='lower right')#,label_formatter = lambda x, y:'')
    ax.add_artist(scalebar)
    
    cbar = plt.colorbar(cax, ax=ax,shrink=0.8,ticks=[clow,chigh])
    cbar.set_ticklabels([str(clow)+" nm", str(chigh)+" nm"])
    cbar.ax.tick_params(labelsize=15)
    return fig,ax

def PlotGrayImage(fig, 
                  gs,
                  image):
    '''
    Plotting image with gray colors

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. subplot inside figure grid (gridspec.GridSpec[0,0])
    image : numpy.ndarray
        DESCRIPTION. image

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. Figure
    ax : matplotlib.axes._axes.Axes
        DESCRIPTION. Axes of figure

    '''
    ax = fig.add_subplot(gs)
    ax.imshow(image, cmap="Greys_r")
    ax.axis('off')
    return fig,ax

def PlotImageHistogram(fig,
                       gs,
                       im,
                       clow=0,
                       chigh=30,
                       colorMap="Grays_r",
                       coordInlet=None):
    '''
    Plot 2d figure and histogram

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. subplot inside figure grid (gridspec.GridSpec[0,0])
    im : numpy.ndarray
        DESCRIPTION. image
    clow : float
        DESCRIPTION. lower intensity limit in plotting
        The default is 0.
    chigh : float
        DESCRIPTION. higher intensity limit in plotting
        The default is 30.
    colorMap : str
        DESCRIPTION. standard cmap name in plt.imshow
        The default is "Grays_r".
    coordInlet : list (or tuple)
        DESCRIPTION. list of subregion of the original image x1,x2,y1,y2
        The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    axHis : matplotlib.axes._axes.Axes
        DESCRIPTION. histogram axes

    '''
    gsgs = gridspec.GridSpecFromSubplotSpec(2, 2,subplot_spec=gs,width_ratios=[9,1],height_ratios= [3, 1])

    axIm = fig.add_subplot(gsgs[0, 0])
    axHis = fig.add_subplot(gsgs[1, 0])
    axBar = fig.add_subplot(gsgs[0, 1])
    
    axHis.hist(im.ravel(),bins=100,density=True,color = "0.7")
    axHis.set_xlabel('Intenisty',fontsize=12.5)
    axHis.set_ylabel('Frequency',fontsize=15.625)
    axHis.set_ylim([0, 0.2])
    plt.setp(axHis.get_yticklabels(), fontsize=12.5);
    plt.setp(axHis.get_xticklabels(), fontsize=12.5);
    
    cax = axIm.imshow(im,colorMap)
    cax.set_clim(clow,chigh)
    axIm.axis('off')
    
    axBar.axis('off')
    cbar = plt.colorbar(cax, ax=axBar,shrink=1,aspect=40,ticks=[clow,chigh])
    cbar.set_ticklabels([str(clow)+"nm", str(chigh)+"nm"])
    cbar.ax.tick_params(labelsize=12.5)
    
    if type(coordInlet)==list and len(coordInlet)==4:
        axins = axIm.inset_axes([0.42, 0.25, 0.8, 0.7]) #this should be written as function of coordInlet
        axins.imshow(im,vmin=clow,vmax=chigh,cmap='afmhot', origin="lower")
        axins.set_xlim(coordInlet[0], coordInlet[1])
        axins.set_ylim(coordInlet[2], coordInlet[3])
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axIm.indicate_inset_zoom(axins, edgecolor="white",linewidth=3)
    return fig, axHis

def PlotBimodalFit(ax,
                   params,
                   im):
    '''
    Plots fit using fitted parameters [mu1,std1,I1,mu2,std2,I3] of a summed gaussian 

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        DESCRIPTION. histogram over which the gaussians will be fitted
    params : Array of float64
        DESCRIPTION. (6,) parameters of bimodal fit [mu1,std1,I1,mu2,std2,I3]
    im : numpy.ndarray
        DESCRIPTION. image 

    '''
    y,x,_=plt.hist(im.ravel(),bins=100,density=True,color = "0.7")
    x=(x[1:]+x[:-1])/2 # correct hist data
    def gauss(x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
        return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
    ax.plot(x,bimodal(x,*params),color='k',lw=3,label='model')
    ax.plot(x,gauss(x,mu=0,sigma=params[1],A=params[2]),'b--',linewidth=2)
    ax.plot(x,gauss(x,mu=params[3]-params[0],sigma=params[4],A=params[5]),'r--',linewidth=2)

def PlotAfmProfilePlot(fig,
                       gs,
                       im,
                       shift,
                       delta,
                       start,
                       end,
                       widthpx=512,
                       widthnm=10000):
    '''
    Makes a zoom of an image and plots a profile line using the start and end coordinates
     _____________
    |            |        ↑xshift
    |       ____ |        ↓
    |      |………| |↕xstart ↑delta
    |______|___|_|        ↓

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. subplot inside figure grid (gridspec.GridSpec[0,0])
    im : numpy.ndarray
        DESCRIPTION. 
    shift : tuple
        DESCRIPTION. size 2 tuple (xshift,yshift) for starting coordinates of zoom
    delta : int
        DESCRIPTION. pixel size of the zoom. xshift+delta<im.shape[0] and y shift+delta<im.shape[1]
    start : tuple
        DESCRIPTION.size 2 tuple (xstart,ystart) for start of profile line
    end : tuple
        DESCRIPTION.size 2 tuple (xend,yend) for end of profile line
    widthpx : float
        DESCRIPTION. known lenght of image in pixels to calculate image scale
        The default is 512.
    widthnm : float
        DESCRIPTION. known lenght of image in pixels to calculate image scale
        The default is 10000.

    '''
    im = im[shift[0]:shift[0]+delta,shift[1]:shift[1]+delta]
    profile = profile_line(im, start, end, linewidth=1, mode='constant')
    
    gsgs = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=gs,width_ratios=[1,1])
    ax1 = fig.add_subplot(gsgs[0, 0])
    ax2 = fig.add_subplot(gsgs[0, 1]) 
    
    cax1=ax1.imshow(im,'afmhot')
    cax1.set_clim(0, 20)
    ax1.axis('off')
    scalebar = ScaleBar(widthnm/widthpx, "nm",length_fraction=0.3,width_fraction=1/30,color='w',frameon=False,location='lower right')#,label_formatter = lambda x, y:'')
    ax1.add_artist(scalebar)
    cbar1 = plt.colorbar(cax1, ax=ax1,shrink=0.75,ticks=[0,20])
    cbar1.set_ticklabels(["0 nm", "20 nm"])
    cbar1.ax.tick_params(labelsize=15)
    ax1.plot([start[1],end[1]],[start[0],end[0]], 'b-', lw=2)

    ax2.plot(profile)
    ax2.set_xlabel('x [nm]',fontsize=15)
    ax2.set_ylabel('Height [nm]',fontsize=15)

def PlotHSB(fig,
            gs,
            orientations,
            im,
            coordInlet=None):
    '''
    Plots orientation as Hue, coherency as saturation and original image as value

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. figure grid
    orientations : dict
        DESCRIPTION. orientations dict with numpy.ndarray for the ['theta'],['coherency'],['energy'] 
    im : numpy.ndarray
        DESCRIPTION. image
    coordInlet : list (or tuple)
        DESCRIPTION. list of subregion of the original image x1,x2,y1,y2
        The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. Figure
    ax : matplotlib.axes._axes.Axes
        DESCRIPTION. Axes of figure

    '''
    # Alternative composition, start as HSV
    imageDisplayHSV = np.zeros((orientations["theta"].shape[0], orientations["theta"].shape[1], 3), dtype=float)
    # Hue is the orientation (nice circular mapping)
    imageDisplayHSV[:, :, 0] = (orientations["theta"] + 90) / 180
    # Saturation is coherency
    imageDisplayHSV[:, :, 1] = orientations["coherency"] / orientations["coherency"]
    # Value is original image
    imageDisplayHSV[:, :, 2] = im / 255

    gsgs = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec=gs,width_ratios=[9,1])

    axIm = fig.add_subplot(gsgs[0, 0])
    axBar = fig.add_subplot(gsgs[0, 1])
    
    axIm.imshow(matplotlib.colors.hsv_to_rgb(imageDisplayHSV))
    
    colorMap = matplotlib.cm.hsv
    colorNorm = matplotlib.colors.Normalize(vmin=-90, vmax=90)
    axBar.axis('off')
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm= colorNorm, cmap=colorMap), ax=axBar,shrink=1,aspect=40,ticks=[-90,90])
    
    if type(coordInlet)==list and len(coordInlet)==4:
        axins = axIm.inset_axes([0.42, 0.35, 0.58, 0.7]) #this should be written as function of coordInlet
        axins.imshow(matplotlib.colors.hsv_to_rgb(imageDisplayHSV), origin="lower")
        axins.set_xlim(coordInlet[0], coordInlet[1])
        axins.set_ylim(coordInlet[2], coordInlet[3])
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axIm.indicate_inset_zoom(axins, edgecolor="white",linewidth=3)
    
    return fig, axIm


def Circular_hist(ax, x, w=None, bins=16, density=True, offset=0, gaps=True):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins,weights=w)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches

def PlotOrientationHistogram(fig,
                             gs,
                             orientations,
                             weighted = True,
                             energyT = 0.02,
                             plotWeightMap =False
                             ):
    '''
    ................

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        DESCRIPTION. figure
    gs : matplotlib.gridspec.SubplotSpec
        DESCRIPTION. figure grid
    orientations : dict
        DESCRIPTION. orientations dict with numpy.ndarray for the ['theta'],['coherency'],['energy'] 
    weighted : bool
        DESCRIPTION. boolean to determine if you want to plot the weighted histogram (with coherency)
        The default is True.
    energyT : float
        DESCRIPTION. energy threshold for weighted histogram (pixels with a lower erngy will not be used for histogram)
        The default is 0.02.
    plotWeightMap : bool
        DESCRIPTION. boolean to determine if you want to plot the map of the weighted histogram
        The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        DESCRIPTION. Figure
    ax : matplotlib.axes._axes.Axes
        DESCRIPTION. Axes of figure

    '''
    ax = fig.add_subplot(gs,projection='polar')
    
    orientationHist = np.pi/180 *orientations['theta']
    normEnergy = (orientations['energy']-np.nanmin(orientations['energy']))/(np.nanmax(orientations['energy'])-np.nanmin(orientations['energy']))
    
    if weighted == False:
        Circular_hist(ax, orientationHist.ravel()[~np.isnan(orientationHist.ravel())], w=None, bins=90,density =True,offset=0,gaps=True)

    if weighted ==True:
        wfilter = orientations['coherency'][normEnergy>energyT]
        orientationHistfilter = orientationHist[(normEnergy>energyT)]
        Circular_hist(ax, orientationHistfilter.ravel()[~np.isnan(orientationHistfilter.ravel())], w=wfilter,  bins=90,density =True,offset=0,gaps=True)
    if plotWeightMap ==True:
        fig2 = plt.figure(tight_layout=True)
        fig2.set_size_inches(5,5) #width, height
        
        ax2 = fig2.add_subplot()
        ax2.imshow(orientations['coherency']*normEnergy>energyT)
    
    return fig, ax