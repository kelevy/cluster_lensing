# importing relevant modules
import numpy as np
import scipy.ndimage
import random
import lensing
import utils
import experiments as exp


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
    specs_dic, corr_noise_bands, rho = exp.specs(experiment)
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


#################################################################################################################################


def cmb_mock_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, nber_ch = 1, cluster_corr_cutouts = None, cl_extragal = None, bl = None, nl = None, nber_obs = 1, lensing_reso = 1):
    
    nx, dx, ny, dy = map_params
    map_params_lensing = [map_params[0]*lensing_reso, map_params[1]/lensing_reso, map_params[2]*lensing_reso, map_params[3]/lensing_reso]
    sims = []
    for i in range(nber_obs):
        sim = utils.make_gaussian_realization(l, cl, map_params_lensing)

        if cluster is not None:        
            M, c, z = cluster
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)), np.random.normal(loc=0.0, scale= centroid_shift_value/(2**0.5))     
            centroid_shift = [x_shift, y_shift]
            grid, _ = utils.make_grid(map_params_lensing, grid_shift = centroid_shift) 
            theta =  np.hypot(grid[0], grid[1]) 
            kappa_map = lensing.NFW_convergence(M, c, z, 1100, theta, dim = 2)
            alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params_lensing)         
            sim = lensing.lens_map(sim, alpha_vec, map_params_lensing) 
            sim = utils.downsample_map(sim, lensing_reso)
        else:       
            sim = utils.downsample_map(sim, lensing_reso)
            
            
        sims_ch_arr = [np.copy(sim) for k in range(nber_ch)]
                
        if cl_extragal is not None:
            if isinstance(cl_extragal, list) is False:
                cl_extragal = [cl_extragal]  
            for j in range(nber_ch):
                extragal_noise_map = utils.make_gaussian_realization(l, cl_extragal[j], map_params)
                sims_ch_arr[j] += extragal_noise_map

        if bl is not None:
            if isinstance(bl, list) is False:
                bl = [bl]
            for j in range(nber_ch):    
                sims_ch_arr[j] = utils.convolve(sims_ch_arr[j], l, np.sqrt(bl[j]), mapparams = map_params)
    
        if cluster_corr_cutouts is not None:
            if np.asarray(cluster_corr_cutouts).ndim == 3:
                cluster_corr_cutouts = [cluster_corr_cutouts]
            nx_cutout, ny_cutout = cluster_corr_cutouts[0][0].shape[0], cluster_corr_cutouts[0][0].shape[1]
            s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
            rand_sel = random.randint(0, len(cluster_corr_cutouts[0])-1)
            rand_ang = random.randint(-180,180)
            for j in range(nber_ch):
                cluster_corr_cutout = scipy.ndimage.rotate(np.nan_to_num(cluster_corr_cutouts[j][rand_sel]), np.nan_to_num(rand_ang), reshape = False, mode = 'reflect')
                sims_ch_arr[j][s:e, s:e] = sims_ch_arr[j][s:e, s:e]+cluster_corr_cutout  
        
        if nl is not None:
            if isinstance(nl, list) is False:
                nl = [nl]
            for j in range(nber_ch):
                noise_map = utils.make_gaussian_realization(l, nl[j], map_params)
                sims_ch_arr[j] += noise_map
        
        sims.append(sims_ch_arr)
    
    if nber_ch == 1 and nber_obs == 1:  
        return sims[0][0]
    
    if nber_ch == 1:
        sims_one_freq = []
        for i in range(len(sims)):
            sims_one_freq.append(sims[i][0])
        return sims_one_freq
    
    if nber_obs == 1:
        return sims[0]
    
    sims_freq_sorted = []
    for i in range(nber_ch):
        maps_at_freq_i = []
        for j in range(nber_obs):
            maps_at_freq_i.append(sims[j][i])
        sims_freq_sorted.append(maps_at_freq_i)
    
    return sims_freq_sorted


def cmb_mock_data_dic(freq_arr, mapparams, l, cl, cluster = None, centroid_shift_value = 0, cluster_corr_cutouts_dic = None, cl_extragal_dic = None, bl_dic = None, nl_dic = None, nber_obs = 1):
    
    nber_ch = len(freq_arr)
    
    if cluster_corr_cutouts_dic is not None:
        cluster_corr_cutouts_arr = [cluster_corr_cutouts_dic[freq] for freq in sorted(cluster_corr_cutouts_dic.keys() )]  
    else:
        cluster_corr_cutouts_arr = None
    
    if cl_extragal_dic is not None:
        cl_extragal_arr = [cl_extragal_dic[freq] for freq in sorted(cl_extragal_dic.keys() )]
    else:
        cl_extragal_arr = None
    
    if bl_dic is not None:
        bl_arr = [bl_dic[freq] for freq in sorted(bl_dic.keys() )]
    else:
        bl_arr = None
    
    if nl_dic is not None:
        nl_arr = [nl_dic[freq] for freq in sorted(nl_dic.keys() )]
    else:
        nl_arr = None
   
    sims = cmb_mock_data(mapparams, l, cl, cluster = cluster, centroid_shift_value = centroid_shift_value, nber_ch = nber_ch, cluster_corr_cutouts = cluster_corr_cutouts_arr, cl_extragal = cl_extragal_arr, bl = bl_arr, nl = nl_arr, nber_obs = nber_obs)
    map_dic = {}
    for i, freq in enumerate(freq_arr):
        map_dic[freq] = sims[i]
   
    return map_dic


def cmb_test_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, cluster_corr_cutouts = None, bl = None, nl = None, nber_obs = 1, estimator_validation = False, noise_comparison = False, clus_positions = False, foreground_bias = False):
    
    if estimator_validation is True:
        sims_clus_arr = []
        kappa_maps = [lensing.NFW(cluster[i][0], cluster[i][1], cluster[i][2], 1100).convergence_map(map_params) for i in range(len(cluster))]
        alpha_vecs = [lensing.deflection_from_convergence(map_params, kappa_maps[i]) for i in range(len(cluster))]
        for i in range(nber_obs):
            sim = utils.make_gaussian_realization(map_params, l, cl) 
            sims_lensed = [lensing.lens_map(map_params, sim, alpha_vecs[i]) for i in range(len(cluster))]
            if bl is not None:
                for j in range(len(sims_lensed)):
                    sims_lensed[j] = utils.convolve(sims_lensed[j], l, np.sqrt(bl), map_params = map_params)
            if nl is not None:
                noise_map =  utils.make_gaussian_realization(map_params, l, nl)
                for j in range(len(sims_lensed)):
                    sims_lensed[j] += noise_map
            sims_clus_arr.append(sims_lensed)
        sims_mass_sorted = []
        for i in range(len(cluster)):
            maps_at_mass_i = []
            for j in range(nber_obs):
                maps_at_mass_i.append(sims_clus_arr[j][i])
            sims_mass_sorted.append(maps_at_mass_i)
        return sims_mass_sorted
    
    
    if noise_comparison is True:
        sims_noise_arr = []
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map) 
        for i in range(nber_obs):
            sim = utils.make_gaussian_realization(map_params, l, cl) 
            sim_lensed = lensing.lens_map(map_params, sim, alpha_vec)
            if bl is not None:
                sim_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), map_params = map_params)
            sims_noise = [np.copy(sim_lensed) for i in range(len(nl))]
            noise_maps = [utils.make_gaussian_realization(map_params, l, nl[i]) for i in range(len(nl))]
            for i in range(len(sims_noise)):
                sims_noise[i] += noise_maps[i]
            sims_noise_arr.append(sims_noise)
        sims_noise_sorted = []
        for i in range(len(nl)):
            maps_at_noise_i = []
            for j in range(nber_obs):
                maps_at_noise_i.append(sims_noise_arr[j][i])
            sims_noise_sorted.append(maps_at_noise_i)
        return sims_noise_sorted
    
    
    if clus_positions is True:
        sims_clus_baseline, sims_clus_centroid_shift = [], []
        kappa_map_baseline = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params)
        alpha_vec_baseline = lensing.deflection_from_convergence(map_params, kappa_map_baseline)
        for i in range(nber_obs):    
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value), np.random.normal(loc=0.0, scale =
                                                                                                         centroid_shift_value) 
            centroid_shift = [x_shift, y_shift]
            kappa_map_centroid_shift = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params,
                                                                                                             centroid_shift)       
            alpha_vec_centroid_shift = lensing.deflection_from_convergence(map_params, kappa_map_centroid_shift)
            sim = utils.make_gaussian_realization(map_params, l, cl)
            sim_lensed_baseline = lensing.lens_map(map_params, sim, alpha_vec_baseline)
            sim_lensed_centroid_shift = lensing.lens_map(map_params, sim, alpha_vec_centroid_shift)
            if bl is not None:
                sim_lensed_baseline = utils.convolve(sim_lensed_baseline, l, np.sqrt(bl), map_params = map_params)
                sim_lensed_centroid_shift = utils.convolve(sim_lensed_centroid_shift, l, np.sqrt(bl), map_params = map_params)
            if nl is not None:
                noise_map =  utils.make_gaussian_realization(map_params, l, nl)
                sim_lensed_baseline += noise_map
                sim_lensed_centroid_shift += noise_map
            sims_clus_baseline.append(sim_lensed_baseline)
            sims_clus_centroid_shift.append(sim_lensed_centroid_shift)           
        return sims_clus_baseline, sims_clus_centroid_shift
      
        
    if foreground_bias is True:
        fname ='../data/mdpl2_cutouts_for_tszksz_clus_detection_M1.7e+14to2.3e+14_z0.6to0.8_15320haloes_boxsize10.0am_dx0.5am.npz'
        cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
        mass_z_key = list(cutouts_dic.keys())[0]
        cutouts = cutouts_dic[mass_z_key]
        scale_fac = fg.compton_y_to_delta_Tcmb(150, uK = True)
        tsz_cutouts, ksz_cutouts  = [], []
        for kcntr, keyname in enumerate( cutouts ):
            tsz_cutout = cutouts[keyname]['y']*scale_fac
            tsz_cutouts.append(tsz_cutout)
            ksz_cutout = cutouts[keyname]['ksz']*random.randrange(-1, 2, 2)
            ksz_cutouts.append(ksz_cutout)
        nx, dx, ny, dy = map_params
        cluster_corr_cutout = ksz_cutouts[0]
        nx_cutout, ny_cutout = cluster_corr_cutout.shape[0], cluster_corr_cutout.shape[1]
        s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
        
        
        sims_clus_baseline, sims_clus_tsz, sims_clus_ksz, sims_clus_tsz_ksz = [], [], [], []
        
        
        M, c, z = cluster    
        centroid_shift = [0, 0]
        grid, _ = utils.make_grid(map_params, grid_shift = centroid_shift) 
        theta =  np.hypot(grid[0], grid[1]) 
        kappa_map = lensing.NFW_convergence(M, c, z, 1100, theta, dim = 2)
        alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params)         
        
        for i in range(nber_obs):
            sim = utils.make_gaussian_realization(l, cl, map_params)
            sim_lensed = lensing.lens_map(sim, alpha_vec, map_params) 
            sim_lensed_baseline, sim_lensed_tsz, sim_lensed_ksz, sim_lensed_tsz_ksz = np.copy(sim_lensed), np.copy(sim_lensed), np.copy(sim_lensed), np.copy(sim_lensed)
            tsz_cutout = utils.rotate(tsz_cutouts[random.randint(0, len(tsz_cutouts)-1)], random.randint(-180,180))
            ksz_cutout = utils.rotate(ksz_cutouts[random.randint(0, len(ksz_cutouts)-1)], random.randint(-180,180))
            tsz_ksz_cutout = tsz_cutout+ksz_cutout
            sim_lensed_tsz[s:e, s:e] = sim_lensed_tsz[s:e, s:e] + tsz_cutout
            sim_lensed_ksz[s:e, s:e] = sim_lensed_ksz[s:e, s:e] + ksz_cutout
            sim_lensed_tsz_ksz[s:e, s:e] = sim_lensed_tsz_ksz[s:e, s:e] + tsz_ksz_cutout
            if bl is not None:
                sim_lensed_baseline = utils.convolve(sim_lensed_baseline, l, np.sqrt(bl), mapparams = map_params)
                sim_lensed_tsz = utils.convolve(sim_lensed_tsz, l, np.sqrt(bl), mapparams = map_params)
                sim_lensed_ksz = utils.convolve(sim_lensed_ksz, l, np.sqrt(bl), mapparams = map_params)
                sim_lensed_tsz_ksz = utils.convolve(sim_lensed_tsz_ksz, l, np.sqrt(bl), mapparams = map_params)
            if nl is not None:     
                noise_map =  utils.make_gaussian_realization(l, nl, map_params)
                sim_lensed_baseline += noise_map
                sim_lensed_tsz += noise_map
                sim_lensed_ksz += noise_map
                sim_lensed_tsz_ksz += noise_map
            sims_clus_baseline.append(sim_lensed_baseline)
            sims_clus_tsz.append(sim_lensed_tsz)
            sims_clus_ksz.append(sim_lensed_ksz)
            sims_clus_tsz_ksz.append(sim_lensed_tsz_ksz)             
        return sims_clus_baseline, sims_clus_ksz, sims_clus_tsz, sims_clus_tsz_ksz
    
    
def cmb_forecast_data(experiment, map_params, l, cl, cluster = None, cl_residual = None, bl = None, nber_obs = 1):    
    nx, dx, ny, dy = map_params        
    if cluster is not None: 
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)   
    
    
    sims_arr = []
    
    for i in range(nber_obs):
        sim = utils.make_gaussian_realization(map_params, l, cl) 
        if cluster is not None: 
            sim = lensing.lens_map(map_params, sim, alpha_vec)
        if cl_residual is not None:
            residual_noise_map = utils.make_gaussian_realization(map_params, l, cl_residual)
            sim += residual_noise_map   
        if bl is not None:  
            sim = utils.convolve(sim, l, np.sqrt(bl), map_params = map_params)
        sims_arr.append(sim)
    
    return sims_arr



def cmb_forecast_test_data(map_params, l, cl, cluster = None, cl_residuals = None, bl = None, nber_obs = 1):
        
    if cluster is not None: 
        kappa_map = lensing.NFW(cluster[0], cluster[1], cluster[2], 1100).convergence_map(map_params) 
        alpha_vec = lensing.deflection_from_convergence(map_params, kappa_map)   
   
    sims_arr_so, sims_arr_fyst, sims_arr_s4wide = [], [], []
    
    for i in range(nber_obs):
        sim = utils.make_gaussian_realization(map_params, l, cl) 
        if cluster is not None: 
            sim = lensing.lens_map(map_params, sim, alpha_vec)
        sim_so, sim_fyst, sim_s4wide = np.copy(sim), np.copy(sim), np.copy(sim)
        if cl_residuals is not None:
            residual_noise_map_so = utils.make_gaussian_realization(map_params, l, cl_residuals[0])
            sim_so += residual_noise_map_so   
            residual_noise_map_fyst = utils.make_gaussian_realization(map_params, l, cl_residuals[1])
            sim_fyst += residual_noise_map_fyst      
            residual_noise_map_s4wide = utils.make_gaussian_realization(map_params, l, cl_residuals[2])
            sim_s4wide += residual_noise_map_s4wide              
        if bl is not None:  
            sim_so = utils.convolve(sim_so, l, np.sqrt(bl), map_params = map_params)
            sim_fyst = utils.convolve(sim_fyst, l, np.sqrt(bl), map_params = map_params)
            sim_s4wide = utils.convolve(sim_s4wide, l, np.sqrt(bl), map_params = map_params)   
        sims_arr_so.append(sim_so)
        sims_arr_fyst.append(sim_fyst)
        sims_arr_s4wide.append(sim_s4wide)
    return sims_arr_so, sims_arr_fyst, sims_arr_s4wide