"""Module to compute experimental quantities.

The module requires that the numpy is installed. Additionally, the module requires the tools file. 

Functions:
    beam_power_spectrum(float, numpy.ndarray) -> numpy.ndarray
    instrumental_noise_power_spectrum(float, numpy.ndarray) -> numpy.ndarray    
    atmospheric_noise_power_spectrum(float, float, float, numpy.ndarray) -> numpy.ndarray
    noise_power_spectrum(foat, float, float, float, numpy.ndarray) -> numpy.ndarray
    specs(str) -> dic, dic, float
    frequencies(str) -> numpy.ndarray
    beam_power_spectrum_dic(str, numpy.ndarray) -> dic
    rebeam(dic) -> numpy.ndarray
    noise_power_spectra_dic(str, numpy.ndarray) -> dic
    
"""


#################################################################################################################################


# Import relevant modules.
import numpy as np
import tools


#################################################################################################################################


def beam_power_spectrum(beam_fwhm, l):
    """Returns the beam power spectrum.
    
    Args:
        beam_fwhm (float): FWHM of the instrumental beam in arcmin.
        l (numpy.ndarray): Multipoles.
        
    Returns:
        bl (numpy.ndarray): Beam power spectrum.
        
    """
    
    fwhm_radians = (np.pi/180)*(beam_fwhm/60)
    sigma = fwhm_radians / np.sqrt(8. * np.log(2.))
    bl = np.exp(-1 * l * (l+1) * sigma ** 2) 
 
    return bl


def instrumental_noise_power_spectrum(noise_val_white, l, beam_fwhm = None):
    """Returns the instrumental/white noise power spectrum.
    
    Args:
        noise_val_white (float): Amplitude of the instrumetal noise power spectrum.
        l (numpy.ndarray): Multipoles.
        beam_fwhm (float, optional): FWHM of the instrumental beam in arcmin used deconvolve the power spectrum.
        
    Returns:
        nl_white (numpy.ndarray): Instrumental noise power spectrum.
        
    """
    
    # Compute white noise power spectrum.
    if l is None:
        l = np.arange(10001)
    n_white = (np.pi/180)*(noise_val_white/60)
    nl_white = np.tile(n_white**2, int(max(l)) + 1 )
    nl_white = np.asarray( [nl_white[int(i)] for i in l] )
    
    # Deconvolve white noise power spectrum.
    if beam_fwhm is not None:
        bl = beam_power_spectrum(beam_fwhm, l)
        nl_white *= bl**(-1)
    
    return nl_white


def atmospheric_noise_power_spectrum(noiseval_red, el_knee, alpha_knee, l, beam_fwhm = None, noiseval_red2 = None, el_knee2 = -1, alpha_knee2 = 0, rho = None):
    """Returns the atmospheric/red auto- or cross-band noise power spectrum.
    
    Args:
        noise_val_red (float): Amplitude of the atmospheric noise power spectrum.
        el_knee (float): Multipole characterising the atmospheric noise power spectrum.
        alpha_knee (float): Exponant characterising the atmospheric noise power spectrum.
        l (numpy.ndarray): Multipoles.
        beam_fwhm (float, optional): FWHM of the instrumental beam in arcmin used deconvolve the power spectrum.
        noise_val_red2 (float, optional): Amplitude of correlated atmospheric noise power spectrum.
        el_knee2 (float, optional): Multipole characterising the correlated atmospheric noise power spectrum.
        alpha_knee2 (float, optional): Exponant characterising the correlated atmospheric noise power spectrum.
        rho (float, optional): Strength of the correlation.        
        
    Returns:
        nl_red (numpy.ndarray): Atmospheric auto- or cross-band noise power spectrum.
        
    """
    
    # Compute red noise power spectrum.
    n_red = ((np.pi/180)*(noiseval_red/60))**2 
    nl_red = n_red*(l/el_knee)**alpha_knee
    nl_red[np.isnan(nl_red)] = 0
    nl_red[np.isinf(nl_red)] = 0
    
    # Compute cross-band red noise power spectrum.
    if noiseval_red2 is not None:
        n_red2 =  ((np.pi/180)*(noiseval_red2/60))**2
        nl_red2= n_red2*(l/el_knee2)**alpha_knee2
        nl_red2[np.isnan(nl_red2)] = 0
        nl_red2[np.isinf(nl_red2)] = 0
        nl_red = rho * (nl_red * nl_red2)**(0.5)
   
    # Deconvolve auto- or cross-band red noise power spectrum.
    if beam_fwhm is not None:
        bl = beam_power_spectrum(beam_fwhm)
        nl_red *= bl**(-1)
    
    return nl_red
 

def noise_power_spectrum(noiseval_white, noiseval_red, el_knee, alpha_knee, l, beam_fwhm = None):
    """Returns the total experimental (instrumental + atmospheric) noise power spectrum.
    
    Args:
        noise_val_white (float): Amplitude of the instrumetal noise power spectrum.
        noise_val_red (float): Amplitude of the atmospheric noise power spectrum.
        el_knee (float): Multipole characterising the atmospheric noise power spectrum.
        alpha_knee (float): Exponant characterising the atmospheric noise power spectrum.
        l (numpy.ndarray): Multipoles.
        beam_fwhm (float, optional): FWHM of the instrumental beam in arcmin used deconvolve the power spectrum.     
        
    Returns:
        nl (numpy.ndarray): Total experimental noise power spectrum.
        
    """
    
    # Compute white noise power spectrum.
    nl_white = instrumental_noise_power_spectrum(noiseval_white, l)
       
    # Add atmospheric noise power spectrum.
    nl_red = atmospheric_noise_power_spectrum(noiseval_red, el_knee, alpha_knee, l)
    nl = nl_white + nl_red
    
    # Deconvolve total noise power spectrum.
    if beam_fwhm is not None:
        bl = beam_power_spectrum(beam_fwhm, l)
        nl *= bl**(-1)
        
    return nl


#################################################################################################################################


def specs(experiment): 
    """Returns the instrumental and atmospheric specifications of the experiment.
    
    Args:
        experiment (str): Name of the experiment. Can be 'spt_3g', 'so', 'so_fyst', or 's4wide'. 
        
    Returns:
       specs_dic (dic): Dictionary containing for each instrumental frequency (float) the corresponding beam value in arcmin 
                        (float), white noise and rednoise levels in uK arcmin (float, float), and paramteres governing the 
                        atmospheric 1/f noise for (float, float).
       corr_noise_bands (dic): Dictionary containing for each intrumental frequency (float) the corresponding correlated 
                               atmospheric noise frequency band (float).
       rho (float): Strength of the correlation.
        
    """ 
    
    if experiment == 'spt3g':
        specs_dic = {
            # freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
            90.:[1.7, 3.0, 3.0,  1200., -3.],
            150.:[1.2, 2.2, 2.2,  2200., -4.], 
            220.:[1.0, 8.8,  8.8,  2300., -4.],
            }
        corr_noise_bands = {90.:[90.], 150.:[150.], 220.:[220.]}
        rho = 1.
     
    if experiment == 'so':
        specs_dic = {
            # freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
            27.:[7.4, 52.1, 6.1,  1000., -3.5],
            39.:[5.1, 27.1, 3.8,  1000., -3.5], 
            93.:[2.2, 5.8,  9.3,  1000., -3.5],
            145.:[1.4, 6.5,  23.8, 1000., -3.5],
            225.:[1.0, 15.0, 80.0, 1000., -3.5],
            280.:[0.9, 37.0, 108.0, 1000., -3.5],
            }
        corr_noise_bands = {27.:[39.], 39.:[27.], 93.:[145.], 145.:[93.], 225.:[280.], 280.:[225.]}
        rho = 0.9
            
    if experiment == 'so_fyst':
        specs_dic = {
            # freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee] 
            27.:[7.4, 52.1, 6.1,  1000., -3.5],
            39.:[5.1, 27.1, 3.8,  1000., -3.5], 
            93.:[2.2, 5.8,  9.3,  1000., -3.5],
            145.:[1.4, 6.5,  23.8, 1000., -3.5],
            220.:[0.95, 14.6, 434.84, 1000., -3.5],
            225.:[1.0, 15.0, 80.0, 1000., -3.5],
            279.999999:[0.9, 37.0, 108.0, 1000, -3.5],
            280.:[0.75, 27.5, 1140.17, 1000., -3.5], 
            350.:[0.58, 104.8, 5648.79, 1000., -3.5],
            410.:[0.50, 376.6, 14174., 1000., -3.5]                           
            }
        corr_noise_bands = {27.:[39.], 39.:[27.], 93.:[145.], 145.:[93.], 220.:[220.], 225.:[279.999999], 279.999999:[225.], 280.:[280.], 350.:[350.], 410.:[410.]}   
        rho = 0.9
    
    if experiment == 's4wide':
        specs_dic = {
            # freq: [beam_arcmins, white_noise, red_noise, elknee, alphaknee]              
            27.:[7.4, 21.5, 21.5, 415., -3.5],
            39.:[5.1, 11.9, 11.9, 391., -3.5], 
            93.:[2.2, 1.9, 1.9, 1932., -3.5],
            145.:[1.4, 2.1, 2.1, 3917., -3.5],
            225.:[0.9, 6.9, 6.9, 6740., -3.5],
            278.:[0.7, 16.8, 16.8, 6792., -3.5],
            }   
        corr_noise_bands = {27.:[39.], 39.:[27.], 93.:[145.], 145.:[93.], 225.:[278.], 278.:[225.]}
        rho = 0.9
        
    return specs_dic, corr_noise_bands, rho


#################################################################################################################################


def frequencies(experiment):
    """Returns the frequencies of the experiment.
    
    Args:
        experiment (str): Name of the experiment. Can be 'spt_3g', 'so', 'so_fyst', or 's4wide'. 
        
    Returns:
        freq_arr (numpy.ndarray): List containing the frequencies of the experiment.
        
    """ 
    
    specs_dic, _, _ = specs(experiment)
    freq_arr = np.array(sorted(specs_dic.keys()))
    
    return freq_arr


def beam_power_spectrum_dic(experiment, l, optimal_beam = None):
    """Returns a dictionary containing for each experimental frequency the corresponding beam power spectrum.
    
    Args:
        experiment (str): Name of the experiment. Can be 'spt_3g', 'so', 'so_fyst', or 's4wide'. 
        l (numpy.ndarray): Multipoles.
        optimal_beam (float, optional) = Includes an optimal beam power spectrum which can be used for rebeaming.
        
    Returns:
        bl_dic (dic): Dictionary containing for each instrumental frequency (float) the corresponding beam power spectrum 
                      (numpy.ndarray).
                        
    """ 
    
    # Compute beam power spectra for all experimental frequencies.
    specs_dic, _, _ = specs(experiment)
    freq_arr =np.array(sorted(specs_dic.keys()))
    bl_dic = {}
    for freq in freq_arr:
        beam_fwhm, noiseval_white, noiseval_red, elknee, alphaknee = specs_dic[freq]
        bl_dic[freq] = beam_power_spectrum(beam_fwhm, l)
    
    # Include optimal beam power spectrum.
    if optimal_beam is not None:
        bl_dic['optimal'] = beam_power_spectrum(optimal_beam, l)
        
    return bl_dic


def rebeam(bl_dic, threshold = 1000):
    """Returns the rebeamed beam power spectrum.
    
    Args:
        bl_dic (dic): Dictionary containing for each instrumental frequency (float) and for the optimal frequency (str) the 
                      corresponding beam power spectra (numpy.ndarray).
        threshold (int, optional): Highest allowed value for the inverse beam power spectrum values.
        
    Returns:
        rebeamed_arr (numpy.ndarray): Array containing the rebeamed beam power spectra.
        
    """
    
    # Extract and sort frequencies. 
    freq_arr = []
    for nu in list(bl_dic.keys()): 
        if isinstance(nu, int) or isinstance(nu, float):
            freq_arr.append(nu)
    freq_arr = sorted(freq_arr)

    # Compute new beam values for each frequency based on the effective beam. 
    bl_eff = bl_dic['optimal']
    rebeamed_arr = np.array([])
    for freq in freq_arr:
        bad_inds = np.where(bl_dic[freq]<0)
        bl_dic[freq][bad_inds] = 0.
        inv_beam_val = 1/bl_dic[freq]
        inv_beam_val[inv_beam_val>threshold] = threshold
        rebeamed_val = bl_eff*inv_beamval
        rebeamed_arr.append(rebeamed_val)
        
    return rebeamed_arr


def noise_power_spectra_dic(experiment, l, deconvolve = False, use_cross_power = False):
    """Returns a dictionary containing for each experimental frequency the corresponding total (instrumental + atmospheric) noise 
       power spectrum.
    
    Args:
        experiment (str): Name of the experiment. Can be 'spt_3g', 'so', 'so_fyst', or 's4wide'. 
        l (numpy.ndarray): Multipoles.
        deconvolve (bool, optional): If True, the total noise power spectra are deconvolved with the corresponding beam.
        use_cross_power(bool, optional): If True, the cross-band noise power spectra are removed.
        
    Returns:
        nl_dic (dic): Dictionary containing for each instrumental frequency (float) the corresponding total noise power spectrum 
                      (numpy.ndarray).
                        
    """ 
    
    # Extract the experiment specifications.
    specs_dic, corr_noise_bands, rho = specs(experiment)
    freq_arr = sorted( specs_dic.keys() )
    beam_arr, whitenoise_arr, rednoise_arr, el_knee_arr, alpha_knee_arr = [], [], [], [], []
    for freq in freq_arr:
        beam_fwhm, noiseval_white, noiseval_red, el_knee, alpha_knee = specs_dic[freq]
        beam_arr.append(beam_fwhm)
        whitenoise_arr.append(noiseval_white)
        rednoise_arr.append(noiseval_red)
        el_knee_arr.append(el_knee)
        alpha_knee_arr.append(alpha_knee) 
      
    # Create a total noise power spectra dictionary.
    nl_dic = {}
    for i in range(len(freq_arr)):
        beam_fwhm1, noiseval_white1, noiseval_red1, el_knee1, alpha_knee1  = beam_arr[i], whitenoise_arr[i], rednoise_arr[i], el_knee_arr[i],  alpha_knee_arr[i]
        for j in range(len(freq_arr)):        
            beam_fwhm2, noiseval_white2, noiseval_red2, el_knee2, alpha_knee2  = beam_arr[j], whitenoise_arr[j], rednoise_arr[j], el_knee_arr[j], alpha_knee_arr[j]
            if freq_arr[i] == freq_arr[j]: 
                if deconvolve is False:
                    nl =  noise_power_spectrum(noiseval_white1, noiseval_red1, el_knee1, alpha_knee1, l)
                else:
                    nl =  noise_power_spectrum(noiseval_white1, noiseval_red1, el_knee1, alpha_knee1, l, beam_fwhm1)
            else:
                if freq_arr[j] in corr_noise_bands[freq_arr[i]]: 
                    nl = atmospheric_noise_power_spectrum(noiseval_red1, el_knee1, alpha_knee1, l, noiseval_red2 = noiseval_red2, el_knee2 = el_knee2, alpha_knee2 = alpha_knee2, rho = rho)
                else:
                    nl = np.zeros(len(l))
            nl[l<=10] = 0.
            nl_dic[(freq_arr[i], freq_arr[j])] = nl              
      
    # Remove cross-band noise power spectra.            
    if use_cross_power is False:
        nl_arr = [nl_dic[freq, freq]  for freq in freq_arr]  
        nl_dic = {}
        for i, freq in enumerate(freq_arr):
            nl_dic[freq] = nl_arr[i]
  
    return nl_dic