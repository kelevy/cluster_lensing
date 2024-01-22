# importing relevant modules
import numpy as np
import scipy.ndimage 
import random
from tqdm import tqdm
import utils
import lensing
import cosmo
import stats
import mockobs
import inpaint


#################################################################################################################################


def get_angle_and_magnitude(signal, dx):
    gradient_map = np.nan_to_num(np.gradient(signal, dx))
    gradient_xmap, gradient_ymap = gradient_map[1], gradient_map[0]
    angle_map = np.degrees(np.arctan2(gradient_ymap, gradient_xmap))
    magnitude_map = np.hypot(gradient_xmap, gradient_ymap) 
    return angle_map, magnitude_map
    

def get_aligned_map(map_params, image, image_noiseless = None, cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None):  

    if image_noiseless is None:
        image_noiseless = np.copy(image)
        
    if cl_noise is None:
        l = np.arange(10001)
        cl = np.ones(max(l)+1)
        cl_noise = np.zeros(max(l)+1)
    
    _, dx, _, _ = map_params
    wiener_filter = utils.wiener_filter(cl, cl_noise)
    low_pass_filter = utils.low_pass_filter(l, l_cut)
    gradient_map = utils.convolve(image, l, wiener_filter * low_pass_filter, map_params) 
    gradient_cutout = utils.central_cutout(gradient_map, cutout_size_for_grad_est_am, map_params)
    angle, magnitude = get_angle_and_magnitude(gradient_cutout, dx)
    angle, magnitude_weight = np.median(angle), np.median(magnitude) 
    image_aligned = utils.rotate(image_noiseless, angle)
    
    return image_aligned, magnitude_weight 
    
 
def get_aligned_map_0(map_params, image, image_noiseless = None,
                      cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, 
                      cl_noise = None, inpaint_map = False, bl = None, nl = None, sigma_dic = None,
                      mask_radius_inner = None, mask_radius_outer = None, noofsims = 1000,  mask_inner = 0):  

    if image_noiseless is None:
        image_noiseless = np.copy(image)
        
    if cl_noise is None:
        l = np.arange(10001)
        cl = np.ones(max(l)+1)
        cl_noise = np.zeros(max(l)+1)
    
    if inpaint_map is True:
        #for inpainting
        nx, dx, ny, dy = map_params
        x1,x2 = -nx/2. * dx, nx/2. * dx
        #If 1, the inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
        #get ra, dec or map-pixel grid
        ra = np.linspace(x1,x2, nx) #arcmins
        dec = np.linspace(x1,x2, nx) #arcmins
        ra_grid, dec_grid = np.meshgrid(ra,dec)
        image_inpainted = inpaint.inpainting(image, ra_grid, dec_grid, map_params, l, cl, bl, nl, 
                                             noofsims, mask_radius_inner, mask_radius_outer, 
                                             low_pass_cutoff = 1, mask_inner = mask_inner,
                                             sigma_dic = sigma_dic)
    else:
        image_inpainted = np.copy(image)
    
    _, dx, _, _ = map_params
    wiener_filter = utils.wiener_filter(cl, cl_noise)
    low_pass_filter = utils.low_pass_filter(l, l_cut)
    gradient_map = utils.convolve(image_inpainted, l, wiener_filter * low_pass_filter, map_params) 
    gradient_cutout = utils.central_cutout(gradient_map, cutout_size_for_grad_est_am, map_params)
    angle, magnitude = get_angle_and_magnitude(gradient_cutout, dx)
    angle, magnitude_weight = np.median(angle), np.median(magnitude) 
    image_aligned = utils.rotate(image_noiseless, angle)
    
    return image_aligned, magnitude_weight, angle 
    
    
def get_stack(cutouts, weights = None):
      
    if weights is None:
        weights = np.ones(len(cutouts))
    weights = np.array(weights)
    weighted_cutouts = [cutouts[i]*weights[i] for i in range(len(cutouts))]
    stack = np.sum(weighted_cutouts, axis = 0)/np.sum(weights)
    
    return stack

    
def get_aligned_stack(maps, map_params, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None, use_gradient_weights = True, 
           noise_weights_clus = None, return_angle = False, inpaint_map = False, 
           bl = None, nl = None, sigma_dic = None, mask_radius_inner = None, mask_radius_outer = None,
           noofsims = 1000,  mask_inner = 0):     

    cutouts_aligned = []
    weights_gradient = []
    angle_arr = []
    for img in maps:
        map_aligned, weight_gradient, angle = get_aligned_map_0(map_params, img, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l_cut = l_cut, 
                                                                l = l, cl = cl, cl_noise = cl_noise, 
                                                                inpaint_map = inpaint_map, bl = bl,
                                                                nl = nl, sigma_dic = sigma_dic,
                                                                mask_radius_inner = mask_radius_inner,
                                                                mask_radius_outer = mask_radius_outer,
                                                                noofsims = noofsims,  mask_inner = mask_inner)
        cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params)
        cutouts_aligned.append(cutout_aligned-np.median(cutout_aligned))
        weights_gradient.append(weight_gradient)
        angle_arr.append(angle)

    if use_gradient_weights is False:
        weights_gradient = np.ones(len(maps))
    if noise_weights_clus is None:
        noise_weights_clus = np.ones(len(maps))  
    
    stack_aligned = get_stack(cutouts_aligned, weights_gradient) 
    if return_angle is False:
        return stack_aligned, cutouts_aligned, weights_gradient   
    else:
        print('returning angles')
        return stack_aligned, cutouts_aligned, weights_gradient, angle_arr
    

def get_tsz(maps, map_params, weights_gradient = None, cutout_size_am = 10, use_gradient_weights = True, 
            noise_weights_clus = None, avg = 20):
     
    stacks_tsz_all = []
    for i in range(avg):
        cutouts_rot = []
        for j, img in enumerate(maps):
            rand_angl = np.random.random() * 360.
            #map_rot = utils.rotate(img, rand_angl)
            map_rot = scipy.ndimage.interpolation.rotate(img, rand_angl, reshape=False, mode='reflect')
            cutout_rot = utils.central_cutout(map_rot, cutout_size_am, map_params)
            cutouts_rot.append(cutout_rot-np.median(cutout_rot))
        if weights_gradient is None:
            weights_gradient = np.ones(len(maps))
            print('Using ', weights_gradient)
        stack_tsz = get_stack(cutouts_rot, weights_gradient)    
        stacks_tsz_all.append(stack_tsz)
    stack_tsz_mean = np.mean(stacks_tsz_all, axis = 0)
     
    #stack_tsz_fit = fit_for_tsz(stack_tsz_mean, map_params[1], cluster_rad_am = 1.5) 
     
    return stack_tsz_mean, cutouts_rot   #stack_tsz_fit
   
     
   
def covariance_and_correlation_matrix(cutouts_aligned_clus, weights_gradient_clus, map_params, cutout_size_am = 10, 
                                      cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None, 
                                      use_gradient_weights = True, noise_weights_clus = None, correct_for_tsz = False,
                                      stack_bg = None, stack_tsz = None):
    
    
    howmany_jk_samples = len(cutouts_aligned_clus)
    sims_for_covariance = []#np.zeros( (npixels, howmany_jk_samples) )
    indces = stats.jackknife_resampling(len(cutouts_aligned_clus), howmany_jk_samples)
    simarr=np.arange(howmany_jk_samples)    
    for n in range(len(simarr)):
        
        curr_cutouts_clus, curr_weights_clus = np.asarray(cutouts_aligned_clus)[indces[n]], np.asarray(weights_gradient_clus)[indces[n]]
        stack_aligned = get_stack(curr_cutouts_clus, weights = curr_weights_clus) 
        
        if stack_tsz is None:
            stack_dipole = stack_aligned - stack_bg
            sims_for_covariance.append(stack_dipole)
                    
        else:
            stack_dipole = stack_aligned - stack_bg - stack_tsz      
            sims_for_covariance.append(stack_dipole)
       
    covariance_matrix, correlation_matrix = stats.covariance_and_correlation_matrix(sims_for_covariance, jackknife = True)
    return covariance_matrix, correlation_matrix


"""
#def perform_simple_jackknife_sampling(total, howmany_jk_samples):
#    each_split_should_contain=int(total * 1./howmany_jk_samples)
#    fullarr=np.arange(total)
#    inds_to_pick=np.arange(len(fullarr))
#    already_picked_inds=[]
#    jk_samples=[]
#    for n in range(howmany_jk_samples):
#        inds=np.random.choice(inds_to_pick, size=each_split_should_contain, replace=0)
#        inds_to_delete=np.where (np.in1d(inds_to_pick, inds) == True)[0]
#        inds_to_pick=np.delete(inds_to_pick, inds_to_delete)
        #push all on the non inds dic into - because for each JK we will ignore the files for this respective sim
#        tmp=np.in1d(fullarr, inds)
#        non_inds=np.where(tmp == False)[0]
#        jk_samples.append( (non_inds) )
#    return np.asarray( jk_samples )

def get_jk_covariance(cutouts, howmany_jk_samples, weights=None, T_or_Q_or_U='T'):

    total_clusters=len(cutouts)
    jk_samples=stats.jackknife_resampling(len(cutouts), howmany_jk_samples)
    #perform_simple_jackknife_sampling(total_clusters, howmany_jk_samples)

    simarr=np.arange(howmany_jk_samples)
    npixels=400

    stacked_cutouts_for_jk_cov=np.zeros( (npixels, howmany_jk_samples) )
    for jkcnt, n in enumerate( simarr ):

        #print('JK=%s of %s' %(jkcnt, howmany_jk_samples), end=' ')
  
        

        weighted_cluster_stack_arr=[]
        curr_cutouts, curr_weights=np.asarray(cutouts)[jk_samples[n]], np.asarray(weights)[jk_samples[n]]
        for (c, w) in zip( curr_cutouts, curr_weights ):

            weighted_cluster_stack_arr.append( c * w )
            
        weighted_cluster_stack_arr=np.asarray(weighted_cluster_stack_arr)

        cluster_stack=np.sum( weighted_cluster_stack_arr, axis=0)/np.sum( curr_weights )
        #imshow(cluster_stack, interpolation='bicubic', cmap=cm.RdYlBu); colorbar(); title(n); show(); sys.exit()
        stacked_cutouts_for_jk_cov[:, n]=np.asarray(cluster_stack.flatten()).flatten()
    #subplot(5,5,n+1); imshow(stacked_cutouts_for_jk_cov[:, n].reshape(ny,nx), interpolation='bicubic', cmap=cm.RdYlBu); colorbar(); title(n); 

    #show(); #sys.exit()
    mean=np.mean(stacked_cutouts_for_jk_cov, axis=1)
    for jkcnt, n in enumerate( simarr ):
        stacked_cutouts_for_jk_cov[:, n]=stacked_cutouts_for_jk_cov[:, n] - mean

    #print(stacked_cutouts_for_jk_cov.shape)
    jk_cov=(howmany_jk_samples - 1) * np.cov(stacked_cutouts_for_jk_cov)    

    return jk_cov
"""

"""
def models(nber_fit, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = None, nl = None, cl_noise = None, cutout_size_am = 10, 
           cutout_size_for_grad_est_am = 6, use_magnitude_weights = True, use_noise_weights = False, apply_noise = True, l_cut = 2000, 
           lensing_reso = 1, average = 1, maps_cmb_unlensed = None, maps_noise = None, ksz_cutouts = None):
    
    nx, dx, ny, dy = map_params
    map_params_lensing = [map_params[0]*lensing_reso, map_params[1]/lensing_reso, map_params[2]*lensing_reso, map_params[3]/lensing_reso]
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
    
    mass_int = np.copy(mass_int)*1e14   
    
    models = []
    for k in range(average):
        cutouts_clus_arr = []
        magnitude_weights_clus_arr = [] 
        models_arr = [] 
        sims_background = []
        sims_background_weights = []
        for i in tqdm(range(nber_fit)):

            if maps_cmb_unlensed is None:
                sim = mockobs.cmb_mock_data(map_params_lensing, l, cl, lensing_reso = lensing_reso)
            else: 
                sim = maps_cmb_unlensed[i+k*nber_fit] 
            if maps_noise is None:
                noise_map = utils.make_gaussian_realization(l, nl, map_params) 
            else:
                noise_map = maps_noise[i+k*nber_fit]
            
            sim_background = utils.downsample_map(sim, lensing_reso)
            if bl is not None:
                sim_background = utils.convolve(sim_background, l, np.sqrt(bl), mapparams = map_params)   
            sim_background_noise = np.copy(sim_background)     
            if apply_noise is True:
                sim_background_noise = sim_background_noise + noise_map 
            
            map_aligned, weight_gradient = get_aligned_map(map_params, sim_background_noise, 
                                                           image_noiseless = sim_background, 
                                                           cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                           l = l, cl = cl, 
                                                           cl_noise = cl_noise,
                                                           l_cut = l_cut)
            if use_magnitude_weights is False:
                weight_gradient = 1
            cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params) 
            cutout_aligned = cutout_aligned - np.median(cutout_aligned)
            sims_background.append(cutout_aligned*weight_gradient)
            sims_background_weights.append(weight_gradient)
            
            
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)), np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)) 
            centroid_shift = [x_shift, y_shift]
            grid, _ = utils.make_grid(map_params_lensing, grid_shift = centroid_shift) 
            theta =  np.hypot(grid[0], grid[1]) 
            
            for j in range(len(mass_int)):
                c200c = cosmo.concentration_parameter(mass_int[j], z, 0.674)
                kappa_map = lensing.NFW_convergence(mass_int[j], c200c, z, 1100, theta, dim = 2)
                alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params_lensing)                
                sim_lensed = lensing.lens_map(sim, alpha_vec, map_params_lensing) 
                sim_lensed = utils.downsample_map(sim_lensed, lensing_reso)  
                sim_lensed_noise = np.copy(sim_lensed)
                if ksz_cutouts is not None:
                    rand_sel = random.randint(0, len(ksz_cutouts)-1)
                    sim_lensed_noise = sim_lensed_noise + ksz_cutouts[rand_sel]
                if bl is not None:
                    sim_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), mapparams = map_params)
                    sim_lensed_noise = utils.convolve(sim_lensed_noise, l, np.sqrt(bl), mapparams = map_params)
                #sim_lensed_noise = np.copy(sim_lensed)  
                if apply_noise is True:
                    sim_lensed_noise = sim_lensed_noise + noise_map 

                map_aligned, weight_gradient = get_aligned_map(map_params, sim_lensed_noise, image_noiseless = sim_lensed, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
                if use_magnitude_weights is False:
                    weight_gradient = 1
                cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params) 
                cutout_aligned = cutout_aligned - np.median(cutout_aligned)
                cutouts_clus_arr.append(cutout_aligned*weight_gradient)
                magnitude_weights_clus_arr.append(weight_gradient)
       
        #stack_bg = np.sum(cutouts_clus_arr[0::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[0::len(mass_int)])
        stack_bg = np.sum(sims_background, axis = 0)/np.sum(sims_background_weights)
        
        for i in tqdm(range(len(mass_int))):
            stack_clus = np.sum(cutouts_clus_arr[i::len(mass_int)],axis=0)/np.sum(magnitude_weights_clus_arr[i::len(mass_int)])
            stack_dipole = stack_clus-stack_bg
            models_arr.append(stack_dipole)
  
        models.append(models_arr)


    return np.mean(models, axis = 0)
"""

def models(nber_fit, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = None, nl = None, 
           cl_noise = None, cutout_size_am = 10,cutout_size_for_grad_est_am = 6, 
           use_magnitude_weights = True, use_noise_weights = False, apply_noise = True, l_cut = 2000, 
           lensing_reso = 1, average = 1, maps_cmb_unlensed = None, maps_noise = None, 
           maps_extragal_noise = None,
           ksz_cutouts = None,
           use_unlensed = False):
    
    nx, dx, ny, dy = map_params
    map_params_lensing = [map_params[0]*lensing_reso, map_params[1]/lensing_reso, map_params[2]*lensing_reso, map_params[3]/lensing_reso]
    if cl_noise is None:
        cl_noise = np.zeros(max(l)+1)
    
    mass_int = np.copy(mass_int)*1e14   
    
    models = []
    for k in range(average):
        cutouts_clus_arr = []
        magnitude_weights_clus_arr = [] 
        models_arr = [] 
        sims_background = []
        sims_background_weights = []
        for i in tqdm(range(nber_fit)):

            if maps_cmb_unlensed is None:
                sim = mockobs.cmb_mock_data(map_params_lensing, l, cl, lensing_reso = lensing_reso)
            else: 
                sim = maps_cmb_unlensed[i+k*nber_fit] 
            if nl is None and maps_noise is None:
                hjklmnb = 0
            elif maps_noise is None:
                noise_map = utils.make_gaussian_realization(l, nl, map_params) 
            else:
                noise_map = maps_noise[i+k*nber_fit]
                
                
            sim_background = utils.downsample_map(sim, lensing_reso)     
            sim_background_noise = np.copy(sim_background)  
            if maps_extragal_noise is not None:
                extragal_map = maps_extragal_noise[i+k*nber_fit]
                sim_background_noise = sim_background_noise + extragal_map 
            if bl is not None:
                sim_background = utils.convolve(sim_background, l, np.sqrt(bl), mapparams = map_params)   
                sim_background_noise = utils.convolve(sim_background_noise, l, np.sqrt(bl), mapparams = map_params)  
            if apply_noise is True:
                sim_background_noise = sim_background_noise + noise_map 
            map_aligned, weight_gradient, angle = get_aligned_map_0(map_params, sim_background_noise, image_noiseless = sim_background, 
                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, 
                                                                    cl_noise = cl_noise, l_cut = l_cut)
            if use_magnitude_weights is False:
                weight_gradient = 1
            cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params) 
            cutout_aligned = cutout_aligned - np.median(cutout_aligned)
            sims_background.append(cutout_aligned*weight_gradient)
            sims_background_weights.append(weight_gradient)
            
            
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)), np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)) 
            centroid_shift = [x_shift, y_shift]
            grid, _ = utils.make_grid(map_params_lensing, grid_shift = centroid_shift) 
            theta =  np.hypot(grid[0], grid[1]) 
            sims_lensed = []
            lensed_angles = []
            lensed_weights = []
            
            if use_unlensed is False:
                for j in range(len(mass_int)):
                    c200c = cosmo.concentration_parameter(mass_int[j], z, 0.674)
                    kappa_map = lensing.NFW_convergence(mass_int[j], c200c, z, 1100, theta, dim = 2)
                    alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params_lensing)                
                    sim_lensed = lensing.lens_map(sim, alpha_vec, map_params_lensing) 
                    sim_lensed = utils.downsample_map(sim_lensed, lensing_reso)  
                    sim_lensed_noise = np.copy(sim_lensed)
                    if maps_extragal_noise is not None:
                        extragal_map = maps_extragal_noise[i+k*nber_fit]
                        sim_lensed_noise = sim_lensed_noise + extragal_map 
                    if bl is not None:
                        sim_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), mapparams = map_params)
                        sim_lensed_noise = utils.convolve(sim_lensed_noise, l, np.sqrt(bl), mapparams = map_params)
                    if apply_noise is True:
                        sim_lensed_noise = sim_lensed_noise + noise_map 
                    _, weight_gradient, lensed_angle = get_aligned_map_0(map_params, sim_lensed_noise, image_noiseless = sim_lensed, 
                                                              cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, 
                                                              cl_noise = cl_noise, l_cut = l_cut)     
                    sims_lensed.append(sim_lensed)
                    lensed_angles.append(lensed_angle)
                    lensed_weights.append(weight_gradient)
                    
                for j in range(len(mass_int)): 
                    # Maybe use mean instead of median and also mean/median for weights!
                    map_aligned = utils.rotate(sims_lensed[j], np.median(lensed_angles))
                    if use_magnitude_weights is False:
                        weight_gradient = 1
                    cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params) 
                    cutout_aligned = cutout_aligned - np.median(cutout_aligned)
                    cutouts_clus_arr.append(cutout_aligned*lensed_weights[j])
                    magnitude_weights_clus_arr.append(lensed_weights[j])
            
            else:    
                for j in range(len(mass_int)):
                    c200c = cosmo.concentration_parameter(mass_int[j], z, 0.674)
                    kappa_map = lensing.NFW_convergence(mass_int[j], c200c, z, 1100, theta, dim = 2)
                    alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params_lensing)                
                    sim_lensed = lensing.lens_map(sim, alpha_vec, map_params_lensing) 
                    sim_lensed = utils.downsample_map(sim_lensed, lensing_reso)  
                    if bl is not None:
                        sim_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), mapparams = map_params)
                    map_aligned = utils.rotate(sim_lensed, angle)
                    if use_magnitude_weights is False:
                        weight_gradient = 1
                    cutout_aligned = utils.central_cutout(map_aligned, cutout_size_am, map_params) 
                    cutout_aligned = cutout_aligned - np.median(cutout_aligned)
                    cutouts_clus_arr.append(cutout_aligned*weight_gradient)
                    magnitude_weights_clus_arr.append(weight_gradient)
       
        #stack_bg = np.sum(cutouts_clus_arr[0::len(mass_int)], axis = 0)/np.sum(magnitude_weights_clus_arr[0::len(mass_int)])
        stack_bg = np.sum(sims_background, axis = 0)/np.sum(sims_background_weights)
        
        for i in tqdm(range(len(mass_int))):
            stack_clus = np.sum(cutouts_clus_arr[i::len(mass_int)],axis=0)/np.sum(magnitude_weights_clus_arr[i::len(mass_int)])
            stack_dipole = stack_clus-stack_bg
            models_arr.append(stack_dipole)
  
        models.append(models_arr)


    return np.mean(models, axis = 0)


def fitfunc(p, radius):
    return p[0] + p[1] * ( 1.0 + (radius/p[2]) ** 2. ) ** (0.5 - (1.5 * p[3]) )


def tsz_beta_model_fitting_func(p, p0, radius, data = None, lbounds = None, ubounds = None, fixed = None, return_fit = 0):            

    #fitfunc = lambda p, radius: p[0] + p[1] * ( 1.0 + (radius/p[2]) ** 2. ) ** (0.5 - (1.5 * p[3]) )

    if hasattr(fixed, '__len__'):                
        p[fixed] = p0[fixed]

    if hasattr(lbounds, '__len__'):
        lbounds = np.asarray(lbounds)
        linds = abs(p)<abs(lbounds)
        p[linds] = lbounds[linds]

    if hasattr(ubounds, '__len__'):
        ubounds = np.asarray(ubounds)
        uinds = abs(p)>abs(ubounds)
        p[uinds] = ubounds[uinds]

    if not return_fit:
        return np.ravel(fitfunc(p, radius) - data)
    else:
        return fitfunc(p, radius)
    
    
def fit_for_tsz(cutout, dx, cluster_rad_am = 1.5):
    import scipy.optimize as optimize
    ny, nx = cutout.shape
    x1, x2 = -nx/2. * dx, nx/2. *dx
    x = np.linspace(x1, x2, nx)
    ra_grid, dec_grid = np.meshgrid(x, x)
    radius_am = np.hypot(ra_grid, dec_grid)
    inside_inds = np.where((radius_am<=cluster_rad_am))
    ini_amp = np.mean( cutout[inside_inds] )

    outside_inds = np.where((radius_am>=2*cluster_rad_am))
    ini_bg = np.mean( cutout[outside_inds] )

    ini_theta_core = 1. #arcmin
    ini_beta_value = 1.

    p0 = [ini_bg, ini_amp, ini_theta_core, ini_beta_value]
    minamp, maxamp = ini_amp * 4., ini_amp * 0.5

    lbounds = None#[ini_bg*0.5, minamp, .25, 0.8]
    ubounds = None#[ini_bg*4., maxamp, 1.5, 1.3]

    p1, success = optimize.leastsq(tsz_beta_model_fitting_func, p0[:], args=(p0, radius_am, cutout, lbounds, ubounds))
    tsz_fit = fitfunc(p1, radius_am)

    return tsz_fit