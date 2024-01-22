"""Module containing useful tools for 1d and 2d signals.

The module requires that the numpy and scipy.ndimage packages are installed. 

Functions:
    make_grid(list) -> list, list
    convert_to_2d(umpy.ndarray, umpy.ndarray, list) -> umpy.ndarray
    convolve(numpy.ndarray, numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    wiener_filter(numpy.ndarray, numpy.ndarray) -> numpy.ndarray
    low_pass_filter(numpy.ndarray, float) -> numpy.ndarray
    downsample_map(numpy.ndarray) -> numpy.ndarray
    central_cutout(numpy.ndarray, float, list) -> numpy.ndarray
    rotate(numpy.ndarray, float) -> numpy.ndarray
    radial_profile(numpy.ndarray, float, float, float) -> numpy.ndarray, numpy.ndarray
    power_spectrum(numpy.ndarray, list) -> numpy.ndarray, numpy.ndarray
    make_gaussian_realization(numpy.ndarray, numpy.ndarray, list) -> numpy.ndarray
    
"""


#################################################################################################################################


# Import relevant modules.
import numpy as np
import scipy.ndimage


#################################################################################################################################


def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    input:
    flatskymyapparams = [nx, ny, dx, dy] where ny, nx = flatskymap.shape; and dy, dx are the pixel resolution in arcminutes.
    for example: [100, 100, 0.5, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with dx = dy = 0.5 arcminutes.
    output:
    lx, ly
    """

    nx, dx, ny, dy = flatskymapparams
    dx = np.radians(dx/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, dx ), np.fft.fftfreq( ny, dx ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

def make_grid(mapparams, Fourier = False, grid_shift = None):    
    """Returns map coordinates and the map extent for given map parameters. 
    
    Args:
        mapparams (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        Fourier (bool, optional): If True, the map coordiantes are given in Fourier space.
        grid_shift (array_like, optional): List containing the x and y coordinate shifts in real space:
                                             x_shift (float): Shift in x-direction.
                                             y_shift (float): Shift in y-direction.
        
    Returns:
        grid (array_like): Array containing the coordinate matrices:
                               gridX (numpy.ndarray): x-coordinate matrix.  
                               gridY (numpy.ndarray): y-coordinate matrix.  
        extent (array_like): Array containing the extent of the image:
                                 xmin (float): Lower horizontal limit of the map.
                                 xmax (float): Upper horizontal limit of the map.
                                 ymin (float): Lower vertical limit of the map.
                                 ymax (float): Upper vertical limit of the map.
                           
    """    

    # Create coordinate vectors and their limits based on map parameters in real space or Fourier space.
    nx, dx, ny, dy = mapparams
    if Fourier:
        dx_rad, dy_rad = (np.pi/180)*(dx/60.), (np.pi/180)*(dy/60.)
        x, y = np.fft.fftfreq(nx, dx_rad), np.fft.fftfreq(ny, dy_rad)
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)
        x, y = 2*np.pi*x, 2*np.pi*y
        xmin, xmax = 2*np.pi*xmin, 2*np.pi*xmax
        ymin, ymax = 2*np.pi*ymin, 2*np.pi*ymax   
    else:
        xmin, xmax = -nx*dx/2, nx*dx/2 
        ymin, ymax = -ny*dy/2, ny*dy/2 
        x, y = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)  
    
    # Create coordinate matrices from coordinate vectors.
    grid = np.meshgrid(x, y)
    
    # Shift coordinate matrices by a constant value in x and y direction.
    if grid_shift is not None:
        gridX, gridY = grid 
        x_shift, y_shift = grid_shift
        gridX += x_shift
        gridY += y_shift 
        grid = gridX, gridY
    
    # Define the bounding box in that the image will fill.
    extent = [xmin, xmax, ymin, ymax]
    
    return grid, extent


def convert_to_2d(x, signal1d, mapparams, Fourier = False, grid_shift = None):
    """Returns the two dimensional version of the one dimensional input signal. 
    
    Args: 
        x (array_like): Independent variable where the 1d signal is measured.
        signal1d (array_like): 1d version of the signal.
        mapparams (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        
    Returns:
       
        signal2d (array_like): 2d version of the signal.
                           
    """ 
    
    grid, _ = make_grid(mapparams, Fourier, grid_shift)
    X, Y = grid[0], grid[1]
    R = np.hypot(X, Y)
    signal2d = np.interp(R.flatten(), x, signal1d, right = 0.).reshape(R.shape) 
    
    return signal2d


#################################################################################################################################


def convolve(signal, freq, kernel, mapparams = None):
    """Returns the convolution between an input signal and a given kernel.
    
    Args:
        signal (numpy.ndarray): 1d or 2d input signal.
        freq (numpy.ndarray): Frequencies.
        kernel (numpy.ndarray): 1d convolution kernel.
        mapparams (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        
    Returns:
        signal_convolved (numpy.ndarray): Convolved signal.
                           
    """
    
    signal_fft = np.fft.fft2(signal)
    if mapparams is not None:
        kernel = convert_to_2d(freq, kernel, mapparams, Fourier = True) 
        kernel[np.isnan(kernel)] = 0
    signal_convolved = np.fft.ifft2(kernel*signal_fft).real 
    
    return signal_convolved


def wiener_filter(psd_signal, psd_noise):
    """Returns the Wiener filter for the case of additive white noise and no blurring.
    
    Args: 
        psd_signal (numpy.ndarray): Power spectral density of the signal of interest.
        psd_noise (numpy.ndarray): Power spectral density of the noise.
        
    Returns:
        wiener_filter (numpy.ndarray): Wiener filter.
                           
    """

    wiener_filter = psd_signal/(psd_signal+psd_noise)
    wiener_filter[np.isnan(wiener_filter)] = 0

    return wiener_filter


def low_pass_filter(freq, freq_cut):
    """Returns a sharp low-filter.
    
    Args: 
        freq (numpy.ndarray): Frequencies.
        freq_cut (float): Cutoff frequency. 
        
    Returns:
        low_pass_filter (numpy.ndarray): Low pass filter.
                           
    """
    
    low_pass_filter = np.ones(len(freq))
    low_pass_filter[np.where(freq.astype(int)>freq_cut)] = 0
    
    return low_pass_filter
      

def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):
    """
    filter_type = 0 - low pass filter
    filter_type = 1 - high pass filter
    filter_type = 2 - band pass
    """
    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter

##################################################################################################################################


def downsample_map(data, N=2): #from N.Whitehorn
    ''' original from N.WHitehorn
    '''
    
    height, width=data.shape
    downsampled_map = np.average(np.split(np.average(np.split(data, width // N, axis=1), axis=-1), height // N, axis=1), axis=-1)
    
    return downsampled_map


def central_cutout(image, cutout_size, mapparams):
    """Returns central cutout of a given image.
    
    Args: 
        image (numpy.ndarray): Image from which the central cutout should be extracted.
        cutout_size (float): Size of the cutout in arcmin. 
        mapparams (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        
    Returns:
        central_cutout (numpy.ndarray): Central cutout of the image.
                           
    """
    
    nx, dx, _, _ = mapparams
    nber_pixels = int(cutout_size/dx)
    s, e = int((nx-nber_pixels)/2), int((nx+nber_pixels)/2)
    central_cutout = image[s:e, s:e]     
    
    return central_cutout   


def rotate(cutout, angle_in_deg):

    rotated_map = scipy.ndimage.interpolation.rotate(np.nan_to_num(cutout), np.nan_to_num(angle_in_deg), reshape=False, mode='reflect')
    
    return rotated_map


def radial_profile(image, binmin, binmax, bin_size):
    """Returns the radial profile of a given image.
    
    Args: 
        image (numpy.ndarray): Image from which the radial profile should be extracted.
        bin_min (float): Lower linmit of the first bin.
        bin_max (float): Upper limint of the last bin.
        bin_size (float): Size of the bins.
        
    Returns:
        bin_ctr (numpy.ndarray): Bin centers.
        radial_profile (numpy.ndarray): Radial profile.
        
    """
    
    # Obtain the different radii based on the underlying grid of the image.
    y, x = np.indices((image.shape))
    center = np.array([(x.max()-x.min())/2, (x.max()-x.min())/2])
    radius = np.hypot(x-center[0], y-center[1])
        
    # Construct the radial profile of the image.
    bins = np.arange(binmin, binmax, bin_size)
    print(bins)
    bin_ctr, rad_prf = np.zeros(len(bins)-1), np.zeros(len(bins)-1)
    hit_count = []
    for bin_nbr, bin_val in enumerate(bins[:-1]):
        ind = np.where((radius >= bin_val)&(radius<bin_val+bin_size))
        bin_ctr[bin_nbr] = (bin_val+bin_size/2)
        hits = len(np.where(np.abs(image[ind])>0)[0])
        if hits>0:
            rad_prf[bin_nbr] = np.sum(image[ind])/hits
             
    return bin_ctr, rad_prf


def power_spectrum(image, map_params, image2 = None, binmin = 0, binmax = 10000):
    """Returns the auto- or cross-power spectrum of a given image or iamges.
    
    Args: 
        image (numpy.ndarray): Image from which the power spectrum should be extracted.
        map_params (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        image2 (numpy.ndarray, optional): If give, the cross-power spectrum of the two images is returned.
        bin_min (float, optinal): Lower linmit of the first bin.
        bin_max (float, optional): Upper limint of the last bin.
        
    Returns:
        freq (numpy.ndarray): Frequencies.
        psd (numpy.ndarray): Auto- or cross-power spectral density of the image or images.
        
    """
    
    # Compute bin size used for the power spectrum computation.
    nx, dx, ny, dy = map_params
    dx_rad = np.radians(dx/60.)
    grid, _ = make_grid(map_params, Fourier = True)
    lx, ly = grid
    bin_size = lx.ravel()[1] -lx.ravel()[0]
    
    # Compute auto- or cross-power spectrum and the coreesponding frequencies.
    if image2 is None:
        image_psd = abs( np.fft.fft2(image) * dx_rad)** 2 / (nx * ny)
    else: 
        image_psd = np.fft.fft2(image) * dx_rad * np.conj( np.fft.fft2(image2) ) * dx_rad / (nx * ny)
    freq, psd = radial_profile(image_psd, grid, binmin = binmin, binmax = binmax, bin_size = bin_size)

    return freq, psd 


##################################################################################################################################


def make_gaussian_realization(freq, psd, mapparams):  
    """Returns the Gaussian realization of a given input power spectrum.
    
    Args:
        freq (array_like): Frequencies.
        psd (array_like): Power spectral density.
        mapparams (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin. 
        
    Returns:
        gaussian_realization (array_like): Gaussian realization of the power spectrum.
                           
    """   
    
    # Create random Gaussian realization and its Fourier transform.
    nx, dx, ny, dy = mapparams
    dx_rad, dy_rad = np.radians(dx/60), np.radians(dy/60)
    gauss_map = np.random.randn(nx, ny)/np.sqrt(dx_rad*dy_rad)
    
    # Convolve Gaussian realization with given power spectral density.
    gaussian_realization = convolve(gauss_map, freq, np.sqrt(psd), mapparams = mapparams)
    gaussian_realization -= np.mean(gaussian_realization)

    return gaussian_realization