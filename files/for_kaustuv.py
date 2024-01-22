import numpy as np
import scipy as sp


def make_grid(mapparams, Fourier = False, grid_shift = None):    
    """ Returns map coordinates and the map extent for given map parameters. 
    
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


def kappa2alpha(kappa_map, map_params):
    """ Returns the deflection angle based on a given convergence map.
    
    Args:
        kappa_map (numpy.ndarray): Convergence map.    
        map_params (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin.       
        
    Returns:
        alpha_vec (list): List containing the X and Y components of the deflection angle:
                              alphaX (array_like): X field of the deflection angle in arcmin.    
                              alphaY (array_like): Y field of the deflection angle in arcmin.
        
    """
    
    # Create map grid in Fourier space.
    grid, _ = make_grid(map_params, Fourier = True)
    lX, lY = grid
    l2d = np.hypot(lX, lY)
    
    # Compute deflection angle from convergence map in Fourier space.
    kappa_fft = np.fft.fft2(kappa_map)
    phi_fft = -2. * kappa_fft/l2d**2
    alphaX_fft =  -1j * lX * phi_fft
    alphaY_fft =  -1j * lY * phi_fft
    alphaX_fft[np.isnan(alphaX_fft)] = 0
    alphaY_fft[np.isnan(alphaY_fft)] = 0
    
    # Compute real space deflection angle.
    alphaX = np.degrees(np.fft.ifft2(alphaX_fft).real)*60
    alphaY = np.degrees(np.fft.ifft2(alphaY_fft).real)*60
    alpha_vec = [alphaX, alphaY]
    
    return alpha_vec

    

def lens_map(unlensed_map, alpha_vec, map_params, poly_deg = 5):   
    """ Returns the lensed version of the underlying map based on the given deflection angle.
    
    Args:
        unlensed_map (numpy.ndarray): Input map.
        alpha_vec (list): Array containing the X and Y components of the deflection angle:
                                    alphaX (array_like): X field of the deflection angle in arcmin.    
                                    alphaY (array_like): Y field of the deflection angle in arcmin.                              
        map_params (list): List containing the map parameters: 
                               nx (int): Number of pixels in horizontal direction.
                               dx (float) Pixel resolution in horizontal direction in arcmin.
                               ny (int): Number of pixels in vertical direction.
                               dy (float): Pixel resolution in vertical direction in arcmin.
       poly_deg (int, optional): Degree of the bivariate spline. Default is 5.                      
                               
    Returns:
        lensed_map (numpy.ndarray): Lensed version of the input map.
        
    """
    
    # Create undeflected field.
    grid, _ = make_grid(map_params)  
    betaX, betaY = grid
   
    # Compute deflected field.   
    alphaX, alphaY = alpha_vec
    thetaX = betaX + alphaX
    thetaY = betaY + alphaY
    
    # Compute lensed map through interpolation using bivariate spline approximation.
    interpolate = sp.interpolate.RectBivariateSpline(betaY[:,0], betaX[0,:], unlensed_map, kx = poly_deg, ky = poly_deg)
    lensed_map  = interpolate.ev(thetaY.flatten(), thetaX.flatten()).reshape([len(betaY), len(betaX)]) 
    
    return lensed_map

