""" Module to compute cosmological quantities.

The module requires that the numpy, scipy, and camb packages are installed. 

Functions:
    hubble_parameter(float) -> float
    critical_density(float) -> float
    comoving_distance(float, float) -> float
    comoving_angular_diameter_distance(float, float) -> float
    angular_diameter_distance(float, float) -> float
    luminosity_distance(float, float) -> float
    cmb_power_spectra(numpy.ndarray) -> dict
    concentration_parameter(float, float) -> float

"""


#################################################################################################################################


# Import relevant modules.
import numpy as np
from scipy import integrate
#import camb


#################################################################################################################################


# Define relevant constants.
pi = 3.141592653589793 
c = 299797 # speed of light in km/s
G = 4.3*10**(-9) # gravitational constant in Mpc/M_sun/s^2/km^2


#################################################################################################################################

        
def hubble_parameter(z, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0):
    """ Returns the Hubble parameter for a given redshift.
    
    Args:
        z (float): Redshift.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter.    
        
    Returns:
        H (float): Hubble parameter in km/s/Mpc.
        
    """
        
    H0 = 100*h
    omegam = omegab+omegac
    omegalambda = 1-omegar-omegam-omegak
    Ez = (omegar*(1+z)**4+omegam*(1+z)**3+omegak*(1+z)**2+omegalambda)**0.5
    Hz = H0*Ez
        
    return Hz
    
    
def critical_density(z, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0):
    """ Returns the critical density for a given redshift.
    
    Args:
        z (float): Redshift.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter.      
        
    Returns:
        rho_c (float): critical density in M_sun/Mpc^3.
        
    """
        
    H = hubble_parameter(z, h, omegab,  omegac, omegak)  
    rho_c = (3*H**2)/(8*pi*G) 
        
    return rho_c  


def comoving_distance(z1, z2, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0):
    """ Returns the comoving distance between two objects.
    
    Args:
        z1 (float): Redshift of the first object.
        z2 (float): Redshift of the second object.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter. 
        
    Returns:
        chi (float): Comoving distance in Mpc.
        
    """ 
    
    y, _ = integrate.quad(lambda z: 1/hubble_parameter(z, h, omegab,  omegac, omegak), z1, z2)
    chi = c*y
    
    return chi 


def comoving_angular_diameter_distance(z1, z2, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0): 
    """ Returns the comoving angular distance between two objects.
    
    Args:
        z1 (float): Redshift of the first object.
        z2 (float): Redshift of the second object.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter. 
        
    Returns:
        fk (float): Comoving angular diameter distance in Mpc.
        
    """ 
    
    H0 = 100*h
    DH = c/H0   
    chi = comoving_distance(z1, z2, h, omegab,  omegac, omegak)  
    if omegak > 0: 
        fk = DH/np.sqrt(omegak)*np.sinh(np.sqrt(omegak)*chi/DH)  
    elif omegak == 0: 
        fk = chi  
    else: 
        fk = DH/np.sqrt(abs(omegak))*np.sin(np.sqrt(abs(omegak))*chi/DH) 
            
    return fk 
    
         
def angular_diameter_distance(z1, z2, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0): 
    """ Returns the angular distance between two objects.
    
    Args:
        z1 (float): Redshift of the first object.
        z2 (float): Redshift of the second object.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter. 
        
    Returns:
        Dang (float): Angular diameter distance in Mpc.
        
    """ 
    
    fk = comoving_angular_diameter_distance(z1, z2, h, omegab,  omegac, omegak)  
    Dang = fk/(1+z2)
        
    return Dang   
    
    
def luminosity_distance(z1, z2, h = 0.674, omegar = 0, omegab = 0.049,  omegac = 0.264, omegak = 0): 
    """ Returns the luminosity distance between two objects.
    
    Args:
        z1 (float): Redshift of the first object.
        z2 (float): Redshift of the second object.
        h (float, optional): Dimensionless Hubble parameter.
        omegar (float, optional): Radiation denisty parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter.  
        
    Returns:
        Dlum (float): Luminosity distance in Mpc.
        
    """ 
    
    fk = comoving_angular_diameter_distance(z1, z2)  
    Dlum = (1+z2)*fk
        
    return Dlum

    
##################################################################################################################################   
    
def cmb_power_spectra(l, h = 0.674, omegab = 0.049,  omegac = 0.264, omegak = 0, mnu = 0.06, tau = 0.054, TCMB = 2.72548, As = 2.101e-9, ns = 0.965,  lensed = True):
    """ Returns the CMB TT, EE, BB, and TE power spectrum using the CAMB module (Lewis et al., 2000).   
    
    Args:
        l (numpy.ndarray): Multipoles.
        h (float, optional): Dimensionless Hubble parameter.
        omegab (float, optional): Baryonic matter denisty parameter.
        omegac (float, optional): Cold dark matter denisty parameter.  
        omegak (float, optional): Curvature denisty parameter. 
        mnu (float, optional): Sum of neutrino masses (in eV).
        tau (float, optional): Optical depth.  
        TCMB (float, optional): CMB temperature (in K). 
        As (float, optional): Primordial amplitude.
        ns (float, optional): Scalar spectral index.
        lensed (bool, optional): If True, the large-scale structure lensed power spectra are returned.
        
    Returns:
        power_spectra (dict): Dictionary containing the CMB power spectra:
                                  'TT' (str): CMB temperature power spectrum in muK^2 (numpy.ndarray).
                                  'EE' (str): CMB E-polarization power spectrum in muK^2 (numpy.ndarray).
                                  'BB' (str): CMB B-polarization power spectrum in muK^2 (numpy.ndarray).
                                  'TE' (str): CMB TE cross power spectrum in muK^2 (numpy.ndarray).

    """     
    
    # Compute power spectra Dl.
    if l is None:
        l = np.arange(10001)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0 = 100*h, ombh2 = omegab*h**2, omch2 = omegac*h**2, omk = omegak, mnu = mnu, tau = tau, TCMB = TCMB)
    pars.InitPower.set_params(As = As, ns = ns)
    pars.set_for_lmax(l[-1])
    results = camb.get_results(pars)
    if lensed: 
        powers = results.get_cmb_power_spectra(pars, CMB_unit = 'muK', lmax = l[-1])['lensed_scalar']
    else:
        powers = results.get_cmb_power_spectra(pars, CMB_unit = 'muK', lmax = l[-1])['unlensed_scalar']
    dl_TT = powers[:,0]
    dl_EE = powers[:,1]
    dl_BB = powers[:,2]
    dl_TE = powers[:,3]
    
    # Convert into Cl.
    conv_fact = 2*np.pi/(l*(l+1))
    cl_TT = dl_TT*conv_fact
    cl_EE = dl_EE*conv_fact
    cl_BB = dl_BB*conv_fact
    cl_TE = dl_TE*conv_fact
    power_spectra = {}
    power_spectra['TT'] = cl_TT
    power_spectra['EE'] = cl_EE
    power_spectra['BB'] = cl_BB
    power_spectra['TE'] = cl_TE
    
    return power_spectra

    
##################################################################################################################################


def m500c_m200c(m500cval, zval=0.7, mdefin = '500c', mdefout = '200c'):
    cval = concentration.concentration(m500cval, mdefin, zval)
    retvals = mass_defs.changeMassDefinition(m500cval, cval, zval, mdefin, mdefout , profile='nfw')
    return retvals


def concentration_parameter(M, z, h = 0.674, mass_def = '200c', model = 'NFW'):
    """ Returns the concentration parameter of a halo using the concentration-mass relation from Duffy et al. (2008). 
    
    Args:
        M (float): Halo mass in M_sun.
        z (float): Redshift.
        h (float, optional): Dimensionless Hubble parameter.
        mass_def (str, optional): Mass definition in which the halo mass is given. Can be '200c' (default) or 'vir'. 
        model (str, optional): Model used to describe the halo. Can be 'NFW' (default) or 'Einasto'.
        
    Returns:
        c (float): Concentration paramter of the halo.
        
    """ 
    
    M_pivot = 2.e12/h 
    
    if model == 'NFW':
        if mass_def == '200c':
            A = 5.71
            B = -0.084
            C = -0.47
        elif mass_def == 'vir':
            A = 7.85 
            B = -0.081
            C = -0.71         
    elif model == 'Einasto':
        if mass_def == '200c':
            A = 6.4
            B = -0.108
            C = -0.62
        elif mass_def == 'vir':
            A = 8.82 
            B = -0.106
            C = -0.87

    if M > 0:
            c = A*((M/M_pivot)**B)*(1+z)**C
    else:
            c = 0.

    return c
