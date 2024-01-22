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
