"""Module to compute gravitiational lensing quantities.

The module requires that the numpy and scipy packages are installed. Additionally, the module requires the 'utils' and 'cosmo' files. 

Functions:
    critical_surface_mass_density(float, float) -> float
    deflection_from_convergence(numpy.ndarray, list) -> list
    lens_map (numpy.ndarray, list, list) -> numpy.ndarray
    NFW_convergence (float, float, float, float, numpy.ndarray, int) -> numpy.ndarray

"""


#################################################################################################################################


# Import relevant modules.
import numpy as np
import scipy as sp
import utils
import cosmo


#################################################################################################################################


# Define relevant constants.
pi = 3.141592653589793 
c = 299797 # speed of light in km/s
G = 4.3*10**(-9) # gravitational constant in Mpc/M_sun/s^2/km^2


#################################################################################################################################
      
    
def critical_surface_mass_density(z_l, z_s):
    """Returns the critical_surface_mass_density for based on the redshift of the lens and of the source.
    
    Args:
        z_l (float): Redshift of the lens.
        z_s (float): Redshift of the source.    
        
    Returns:
        sigma_c (float): Critical surface mass density in M_sun/Mpc^2.
        
    """
    
    D_l = cosmo.angular_diameter_distance(0, z_l)  
    D_s = cosmo.angular_diameter_distance(0, z_s)  
    D_ls = cosmo.angular_diameter_distance(z_l, z_s)
    sigma_c = ((c**2)/(4*pi*G))*((D_s)/(D_l*D_ls))  
    
    return sigma_c 
  
   
# Maybe change name of function.  
def deflection_from_convergence(kappa_map, map_params):
    """Returns the deflection angle based on a given convergence map.
    
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
    grid, _ = utils.make_grid(map_params, Fourier = True)
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
    """Returns the lensed version of the underlying map based on the given deflection angle.
    
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
    grid, _ = utils.make_grid(map_params)  
    betaX, betaY = grid
   
    # Compute deflected field.   
    alphaX, alphaY = alpha_vec
    thetaX = betaX + alphaX
    thetaY = betaY + alphaY
    
    # Compute lensed map through interpolation using bivariate spline approximation.
    interpolate = sp.interpolate.RectBivariateSpline(betaY[:,0], betaX[0,:], unlensed_map, kx = poly_deg, ky = poly_deg)
    lensed_map  = interpolate.ev(thetaY.flatten(), thetaX.flatten()).reshape([len(betaY), len(betaX)]) 
    
    return lensed_map


#################################################################################################################################    

def NFW_convergence(M_200c, c_200c, z_l, z_s, theta, dim = 1):
    """Returns the convergence profile or map of an NFW halo according to Bartelmann (1996).
    
    Args:
        M_200c (float): Halo mass based on the 200c mass definition in M_sun.
        c_200c (float): Concentration paramter of the halo based on the 200c mass definition.
        z_l (float): Redshift of the lens.
        z_s (float): Redshift of the source.
        theta (numpy.ndarray): Angular range or grid in arcmin.
        dim (int): Dimension of the convergence.
          
    Returns:
        kappa (numpy.ndarray): NFW convergence profile or map.
            
    """
    
    # Compute kappa_s.   
    rho_c = cosmo.critical_density(z_l)    
    r_200c = ((3*M_200c)/(4*np.pi*200*rho_c))**(1/3)   
    r_s = r_200c/c_200c  
    rho_s = (200./3.)*rho_c*(c_200c**3/(np.log(1+c_200c)-(c_200c/(1+c_200c))))     
    sigma_c = critical_surface_mass_density(z_l, z_s)  
    kappa_s = rho_s*r_s/sigma_c
    
    # Compute x.
    theta_rad = (np.pi/180)*(theta.flatten()/60)
    D_l = cosmo.angular_diameter_distance(0, z_l) 
    theta_s = r_s/D_l
    x = theta_rad/theta_s
        
    # Compute f.
    x1 = np.where(x > 1)
    x2 = np.where(x == 1)
    x3 = np.where(x < 1)
    f = np.zeros(len(x))
    f[x1] = (1/(x[x1]**2-1)**0.5)*np.arctan((x[x1]**2-1)**0.5)
    f[x2] = 1
    f[x3] = (1/(1-x[x3]**2)**0.5)*np.arctanh((1-x[x3]**2)**0.5)
           
    # Compute kappa.
    kappa = (2*kappa_s*(1-f)/(x**2-1)) 
    if dim == 2:
        kappa = kappa.reshape([int(len(kappa)**0.5), int(len(kappa)**0.5)])
        
    return kappa