#%%
print('Importing relevant modules ...')
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import rcParams
import matplotlib
#from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
#rcParams['font.family'] = 'serif'
#rc('text.latex',preamble=r'\usepackage{/Volumes/Extreme_SSD/cluster_lensing/files/apjfonts}')
from mpl_toolkits.axes_grid1 import make_axes_locatable
color_arr = ['olivedrab', 'steelblue', 'darkgoldenrod']
import cosmo
import lensing
import lensing_estimator
import mockobs
import stats
import utils


#################################################################################################################################################
#%%


print('Defining relevant parameters ...')
nber_clus = 2500
nber_rand = 50000
cutout_size_am = 6
cutout_size_for_grad_est_am = 6
l_cut = 2000
average = 1
average_run = 1
nber_runs = 20
nx = 120
ny = 120
dx = 0.5
dy = 0.5
reso_arcmin = 0.5
map_params = [nx, dx, ny, dy]
beam_fwhm = 1.0 # arcmin
noiseval_white_arr = [0.1, 0.5, 1., 3., 5., 7.] # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl_arr = []
cl_noise_arr = []
for noiseval_white in noiseval_white_arr:
    nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
    nl_arr.append(nl)
    cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
    cl_noise_arr.append(cl_noise)

z = 0.7
mass_reso = 0.01
mass_min = 1.5
mass_max = 2.5
mass_int = np.arange(mass_min, mass_max, mass_reso)
M_input = 2e14
use_unlensed = False

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 17
labelsize = 15

# for saving
save_loc_sims_unlensed = '/data/gpfs/projects/punim1922/cluster_lensing/sims/unlensed_cmb'
if not os.path.exists(save_loc_sims_unlensed):
    os.makedirs(save_loc_sims_unlensed)
save_loc_sims_lensed = '/data/gpfs/projects/punim1922/cluster_lensing/sims/lensed_cmb'
if not os.path.exists(save_loc_sims_lensed):
    os.makedirs(save_loc_sims_lensed)
save_loc_sims_noise = '/data/gpfs/projects/punim1922/cluster_lensing/sims/noise_maps'
if not os.path.exists(save_loc_sims_noise):
    os.makedirs(save_loc_sims_noise)
    
save_loc_mockobs_unlensed = '/data/gpfs/projects/punim1922/cluster_lensing/mockobs/unlensed'
if not os.path.exists(save_loc_mockobs_unlensed):
    os.makedirs(save_loc_mockobs_unlensed)
save_loc_mockobs_lensed = '/data/gpfs/projects/punim1922/cluster_lensing/mockobs/lensed'
if not os.path.exists(save_loc_mockobs_lensed):
    os.makedirs(save_loc_mockobs_lensed)
        
save_loc_models = '/data/gpfs/projects/punim1922/cluster_lensing/models'
if not os.path.exists(save_loc_models):
    os.makedirs(save_loc_models)
    

save_loc_results = '/data/gpfs/projects/punim1922/cluster_lensing/results/estimator_comparison'
if not os.path.exists(save_loc_results):
     os.makedirs(save_loc_results)   
    
    
#################################################################################################################################################
#%%


print('Generating unlensed CMB sims ...')
if not os.path.exists(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy'):
    print('for background ...')
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = nber_rand)
    np.save(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy', sim_unlensed)
    del sim_unlensed
if not os.path.exists(save_loc_sims_unlensed+'/60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy'):
    print('for mock data ...')
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = nber_runs*nber_clus)
    np.save(save_loc_sims_unlensed+'/60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy', sim_unlensed)
    del sim_unlensed
if not os.path.exists(save_loc_sims_unlensed+'/60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy'):
    print('for models ...')
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = average*nber_clus)
    np.save(save_loc_sims_unlensed+'/60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy', sim_unlensed)
    del sim_unlensed

print('Generating lensed CMB sims ...')
if not os.path.exists(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy'):
    centroid_shift_value =  0
    x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)), np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)) 
    centroid_shift = [x_shift, y_shift]
    grid, _ = utils.make_grid(map_params, grid_shift = centroid_shift) 
    theta =  np.hypot(grid[0], grid[1]) 
    c200c = cosmo.concentration_parameter(M_input, z, 0.674)
    kappa_map = lensing.NFW_convergence(M_input, c200c, z, 1100, theta, dim = 2)
    alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params)   
    sims_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy')
    sims_lensed = []
    for sim_unlensed in tqdm(sims_unlensed):            
        sim_lensed = lensing.lens_map(sim_unlensed, alpha_vec, map_params) 
        sims_lensed.append(sim_lensed)
    np.save(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(nber_runs*nber_clus)+'lensed_sims.npy', sims_lensed)  
    del sims_unlensed
    del sims_lensed
    
print('Generating white noise sims ...')
for j in range(len(noiseval_white_arr)):
    print('for background ...')
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy'):
        sims_white_noise = []
        for i in tqdm(range(nber_rand)):            
            sim_white_noise = utils.make_gaussian_realization(l, nl_arr[j], map_params) 
            sims_white_noise.append(sim_white_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy', sims_white_noise)
        del sims_white_noise
    print('for lensed mock obs ...') 
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy'):
        sims_white_noise = []
        for i in tqdm(range(nber_runs*nber_clus)):            
            sim_white_noise = utils.make_gaussian_realization(l, nl_arr[j], map_params) 
            sims_white_noise.append(sim_white_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy', sims_white_noise)
        del sims_white_noise
    print('for models ...')
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy'):
        sims_white_noise = []
        for i in tqdm(range(average*nber_clus)):            
            sim_white_noise = utils.make_gaussian_realization(l, nl_arr[j], map_params) 
            sims_white_noise.append(sim_white_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy', sims_white_noise)
        del sims_white_noise 
    
 
###################################################################################################################################
#%%


print('Generating unlensed mock obs  ...')
for j in range(len(noiseval_white_arr)):
    print(noiseval_white_arr[j])
    if not os.path.exists(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy'):
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy')
        sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy')
        mockobs_unlensed = []
        for i in tqdm(range(nber_rand)):
           # sim_cmb_unlensed = #mockobs.cmb_mock_data(map_params, l, cl) #sims_cmb_unlensed[i]
           # sim_white_noise = #utils.make_gaussian_realization(l, nl, map_params) #sims_white_noise[i] 
            mockob_unlensed = utils.convolve(sims_cmb_unlensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]   
            mockobs_unlensed.append(mockob_unlensed)
        np.save(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy', mockobs_unlensed) 
        
        _, extent = utils.make_grid([mockob_unlensed.shape[0], reso_arcmin, mockob_unlensed.shape[1], reso_arcmin])
        fig, ax = plt.subplots(figsize=(5,5))
        title = 'Unlensed Mock Observation'
        im = ax.imshow(mockob_unlensed, extent = extent, cmap = cm.RdBu_r)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax = cax)
        ax.axvline(0, color = 'forestgreen')
        ax.axhline(0, color = 'forestgreen')
        cb.set_label(label=r'$T \ [\mu$'+'K]', fontsize = fontsize-5)
        cb.ax.tick_params(labelsize=labelsize)
        ax.set_xlabel(r'$\theta_x\ $'+'[arcmin]', fontsize = fontsize-5)
        ax.set_ylabel(r'$\theta_y\ $'+'[arcmin]', fontsize = fontsize-5)
        ax.tick_params(labelsize = labelsize)
        ax.set_title(title, fontsize = fontsize-5)
        fig.tight_layout()
        plt.show()
        fig.savefig(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_unlensed_mockob.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
    
        del sims_cmb_unlensed
        del sims_white_noise
        del mockobs_unlensed
    
print('Generating lensed mock obs  ...')
for j in range(len(noiseval_white_arr)):
    print(noiseval_white_arr[j])
    if not os.path.exists(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy'):
        sims_cmb_lensed = np.load(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy') 
        sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
        mockobs_lensed = []
        for i in tqdm(range(nber_runs*nber_clus)):
            mockob_lensed = utils.convolve(sims_cmb_lensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]
            mockobs_lensed.append(mockob_lensed)
        np.save(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[j])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', mockobs_lensed) 
        del sims_cmb_lensed
        del sims_white_noise
        del mockobs_lensed  


###################################################################################################################################
#%%


# generating models
print('Generating models ...')
models_arr = []
for i in range(len(noiseval_white_arr)):
    print(noiseval_white_arr[i])
    
    if not os.path.exists(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy'):
        print('Generating current models ...')
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy')
        sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy')
        models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
                                          cl_noise = cl_noise_arr[i], cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                          l_cut = l_cut, maps_cmb_unlensed = sims_cmb_unlensed, maps_noise = sims_white_noise , average = average,
                                          use_unlensed = use_unlensed)
        np.save(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy', models)
        del sims_cmb_unlensed
        del sims_white_noise
        
    else:
        print('Loading current models ...')
        models = np.load(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy')

    models_arr.append(models)  
    
#%%

for i in range(len(noiseval_white_arr)):
    models_check = [models_arr[i][0], models_arr[i][int(len(mass_int)/2)], models_arr[i][-1]]
    _, extent = utils.make_grid([models_check[0].shape[0], reso_arcmin, models_check[0].shape[1], reso_arcmin])
    fig, axarr = plt.subplots(len(models_check), 1, figsize=(10,10))
    dec = len(str(mass_reso).split(".")[1])
    titles = ['Model '+str(np.round(mass_int[0], dec)), 'Model '+str(np.round(mass_int[int(len(mass_int)/2)], dec)), 'Model '+str(np.round(mass_int[-1], dec))]
    
    for j in range(len(axarr)):
        titstr = titles[j]
        im = axarr[j].imshow(models_check[j], extent = extent, cmap = cm.RdBu_r)
        divider = make_axes_locatable(axarr[j])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = fig.colorbar(im, cax = cax)
        axarr[j].axvline(0, color = 'forestgreen')
        axarr[j].axhline(0, color = 'forestgreen')
        cb.set_label(label=r'$T \ [\mu$'+'K]', fontsize = fontsize-5)
        cb.ax.tick_params(labelsize=labelsize)
        axarr[j].set_xlabel(r'$\theta_x\ $'+'[arcmin]', fontsize = fontsize-5)
        axarr[j].set_ylabel(r'$\theta_y\ $'+'[arcmin]', fontsize = fontsize-5)
        axarr[j].tick_params(labelsize = labelsize)
        axarr[j].set_title(titstr, fontsize = fontsize-5)
    fig.tight_layout()
    plt.show()
    
    fig.savefig(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


###################################################################################################################################
#%%


# generating background stack
print('Generating background stack ...')
for i in range(len(noiseval_white_arr)):
    print(noiseval_white_arr[i])
    if not os.path.exists(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_rand)+'sims.npy'):
        mockobs_unlensed = np.load(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy') 
        stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(mockobs_unlensed, map_params, l = l, cl = cl, 
                                                                                        cl_noise = cl_noise_arr[i], cutout_size_am = cutout_size_am, 
                                                                                        cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                        l_cut = l_cut)  
        del mockobs_unlensed
        np.save(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
    else:
        stack_bg = np.load(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(nber_rand)+'sims.npy') 

    print('Cheking background stack ...')
    _, extent = utils.make_grid([stack_bg.shape[0], reso_arcmin, stack_bg.shape[1], reso_arcmin])
    fig, ax = plt.subplots(figsize=(5,5))
    titstr = 'Background Stack'
    im = ax.imshow(stack_bg, extent = extent, cmap = cm.RdBu_r)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax = cax)
    ax.axvline(0, color = 'forestgreen')
    ax.axhline(0, color = 'forestgreen')
    cb.set_label(label=r'$T \ [\mu$'+'K]', fontsize = fontsize-5)
    cb.ax.tick_params(labelsize=labelsize)
    ax.set_xlabel(r'$\theta_x\ $'+'[arcmin]', fontsize = fontsize-5)
    ax.set_ylabel(r'$\theta_y\ $'+'[arcmin]', fontsize = fontsize-5)
    ax.tick_params(labelsize = labelsize)
    ax.set_title(titstr, fontsize = fontsize-5)
    fig.tight_layout()
    plt.show()
    
    fig.savefig(save_loc_results+'/background_stack_noiseval'+str(noiseval_white_arr[i])+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

###################################################################################################################################
#%%

print('Starting analysis ...')


x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
median_mass_arr = []
median_error_arr = []
pvalue_arr = []

maps = []
for i in range(len(noiseval_white_arr)):
    print(noiseval_white_arr[i])
    mockobs_lensed_baseline = np.load(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white_arr[i])+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', allow_pickle = True) 
    maps.append(mockobs_lensed_baseline) 

stacks_dipole_all_arr = []
covariance_matrices_all_arr = []


for i in tqdm(range(nber_runs)):    
    x_arr_curr = []
    L_arr_curr = []
    M_ip_arr_curr = []
    L_ip_arr_curr = []
    median_mass_arr_curr = []    
    median_error_arr_curr = []
    pvalue_arr_curr = []
    
    print('Computing cluster stacks ...')
    stacks_aligned, cutouts_aligned_arr, weights_gradient_arr = [], [], []
    for  j in range(len(maps)):
        stack_aligned, cutouts_aligned, weights_gradient = lensing_estimator.get_aligned_stack(maps[j][i*nber_clus:i*nber_clus+nber_clus], 
                                                                                               map_params, 
                                                                                               l = l, cl = cl, 
                                                                                               cl_noise = cl_noise_arr[j], 
                                                                                               cutout_size_am = cutout_size_am, 
                                                                                               cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                               l_cut = l_cut)  
        stacks_aligned.append(stack_aligned)
        cutouts_aligned_arr.append(cutouts_aligned)
        weights_gradient_arr.append(weights_gradient)
                                                            

    print('Computing dipole stacks ...')
    stacks_dipole = []
    for j in range(len(maps)):
        stack_dipole = stacks_aligned[j] - stack_bg 
        stacks_dipole.append(stack_dipole)
    stacks_dipole_all_arr.append(stacks_dipole)       
   
    
    print('Computing covariance matrices ...')
    covariance_matrix_arr, correlation_matrix_arr = [], []
    for j in range(len(maps)):  
        covariance_matrix, correlation_matrix = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned_arr[j], 
                                                                                                    weights_gradient_arr[j], 
                                                                                                    map_params, 
                                                                                                    cutout_size_am = cutout_size_am, 
                                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                    l_cut = l_cut, l = l, cl = cl, 
                                                                                                    cl_noise = cl_noise_arr[j], 
                                                                                                    stack_bg = stack_bg,
                                                                                                    stack_tsz = None)
        covariance_matrix_arr.append(covariance_matrix)
        correlation_matrix_arr.append(correlation_matrix)
        
    covariance_matrices_all_arr.append(covariance_matrix_arr)
    
    print('Getting likelihoods and mass estimates ...')
    print('Run', i+1)
    for j in range(len(stacks_dipole)):  
        x, L, median_value, error = stats.run_ml(stacks_dipole[j], models_arr[j], covariance_matrix_arr[j], mass_int, finer_reso = False)
        x_arr_curr.append(x)
        L_arr_curr.append(L)
        print(median_value, error) 
        M_ip, L_ip, median_value_finer, error_finer = stats.run_ml(stacks_dipole[j], models_arr[j], covariance_matrix_arr[j], mass_int, finer_reso = True)
        M_ip_arr_curr.append(M_ip)
        L_ip_arr_curr.append(L_ip)
        median_mass_arr_curr.append(median_value_finer)
        median_error_arr_curr.append(error_finer)
        print(median_value_finer, error_finer) 
        
        
        import scipy as sp
        sp.stats.chisqprob = lambda chisq, df: sp.stats.chi2.sf(chisq, df)
        print('median_value: ', median_value)
        print('mass interval: ', mass_int[np.argmax(L)])
        diff_vector = stacks_dipole[j].flatten() - models[np.argmax(L)].flatten()
        cov_inv = sp.linalg.pinv(covariance_matrix_arr[j])
        chisqval = np.asarray( np.dot(diff_vector.T, np.dot(cov_inv , diff_vector ))).squeeze() 
        pvalue = sp.stats.chisqprob(chisqval,len(diff_vector)-1) 
        pvalue_arr_curr.append(pvalue)
        print(pvalue)
        
     
    x_arr.append(x_arr_curr)
    L_arr.append(L_arr_curr)
    M_ip_arr.append(M_ip_arr_curr)
    L_ip_arr.append(L_ip_arr_curr)
    median_mass_arr.append(median_mass_arr_curr)
    median_error_arr.append(median_error_arr_curr)
    pvalue_arr.append(pvalue_arr_curr)

np.save(save_loc_results+'/all_stacked_dipoles.npy', stacks_dipole_all_arr)
np.save(save_loc_results+'/all_covariance_matrices.npy', covariance_matrices_all_arr)
np.save(save_loc_results+'/pvalues.npy', pvalue_arr)
 
#%%

print('Sorting individual runs after case ...')
L_arr_sorted = []
L_ip_arr_sorted = []
median_mass_arr_sorted = []
median_error_arr_sorted = []
for i in range(len(stacks_dipole)):
    L_arr_sorted_curr = []
    L_ip_arr_sorted_curr = []
    median_mass_arr_sorted_curr = []
    median_error_arr_sorted_curr = []
    for j in range(nber_runs):
        L_arr_sorted_curr.append(L_arr[j][i])
        L_ip_arr_sorted_curr.append(L_ip_arr[j][i]) 
        median_mass_arr_sorted_curr.append(median_mass_arr[j][i]) 
        median_error_arr_sorted_curr.append(median_error_arr[j][i]) 
    L_arr_sorted.append(L_arr_sorted_curr)
    L_ip_arr_sorted.append(L_ip_arr_sorted_curr)
    median_mass_arr_sorted.append(median_mass_arr_sorted_curr)
    median_error_arr_sorted.append(median_error_arr_sorted_curr)
    
print('Getting combined likelihoods and mass estimates ...')  
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(stacks_dipole)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int, L_arr_sorted[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    comb_L_arr.append(L_comb)
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int, L_arr_sorted[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
   

np.save(save_loc_results+'/likelihoods.npy', L_arr_sorted)
np.save(save_loc_results+'/likelihoods_finer.npy', L_ip_arr_sorted)
np.save(save_loc_results+'/comb_likelihoods.npy', comb_L_arr)
np.save(save_loc_results+'/comb_likelihoods_finer.npy', comb_L_ip_arr)
np.save(save_loc_results+'/mass_results.npy', median_mass_arr_sorted)
np.save(save_loc_results+'/error_results.npy', median_error_arr_sorted)
np.save(save_loc_results+'/comb_median_masses.npy', comb_median_mass_arr)
np.save(save_loc_results+'/comb_mass_errors.npy', comb_error_arr)


combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
    
    
file = open(save_loc_results+'/mass_results.txt',"w")
file.write(combined_median_masses_results_file)
file.close()


###################################################################################################################################
#%%

print('Checking dipoles ...')
dipole_checks = [stacks_dipole[0], stacks_dipole[1], stacks_dipole[2], 
                 stacks_dipole[3], stacks_dipole[4], stacks_dipole[5]]

_, extent = utils.make_grid([dipole_checks[0].shape[0], reso_arcmin, dipole_checks[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(dipole_checks), 1, figsize=(20,20))
titles = [str(noiseval_white_arr[0]), str(noiseval_white_arr[1]), str(noiseval_white_arr[2]), 
          str(noiseval_white_arr[3]), str(noiseval_white_arr[4]), str(noiseval_white_arr[5])]

for i in range(len(axarr)):
    titstr = titles[i]
    im = axarr[i].imshow(dipole_checks[i], extent = extent, cmap = cm.RdBu_r)
    divider = make_axes_locatable(axarr[i])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(im, cax = cax)
    axarr[i].axvline(0, color = 'forestgreen')
    axarr[i].axhline(0, color = 'forestgreen')
    cb.set_label(label=r'$T \ [\mu$'+'K]', fontsize = fontsize-5)
    cb.ax.tick_params(labelsize=labelsize)
    axarr[i].set_xlabel(r'$\theta_x\ $'+'[arcmin]', fontsize = fontsize-5)
    axarr[i].set_ylabel(r'$\theta_y\ $'+'[arcmin]', fontsize = fontsize-5)
    axarr[i].tick_params(labelsize = labelsize)
    axarr[i].set_title(titstr, fontsize = fontsize-5)
fig.tight_layout()
plt.show()

fig.savefig(save_loc_results+'/dipoles_check.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  


#%%
# checking covariane matrices
#ksz_check = np.mean(ksz_cutouts[:2500], axis = 0)[50:70, 50:70]
#tsz_check = np.mean(tsz_cutouts[:2500], axis = 0)[50:70, 50:70]
#ksz_tsz_check = [ksz_check, tsz_check]
#_, extent = utils.make_grid([ksz_tsz_check[0].shape[0], reso_arcmin, ksz_tsz_check[0].shape[1], reso_arcmin])
#fig, axarr = plt.subplots(len(ksz_tsz_check), 1, figsize=(8,8))
#titles = ['kSZ Stack', 'tSZ Stack']

#for i in range(len(axarr)):
#    titstr = titles[i]
#    im = axarr[i].imshow(ksz_tsz_check[i], extent = extent, cmap = cm.RdBu_r)
#    divider = make_axes_locatable(axarr[i])
#    cax = divider.append_axes("right", size="5%", pad=0.1)
#    cb = fig.colorbar(im, cax = cax)
#    axarr[i].axvline(0, color = 'forestgreen')
#    axarr[i].axhline(0, color = 'forestgreen')
#    cb.set_label(label=r'$T \ [\mu$'+'K]', fontsize = fontsize-5)
#    cb.ax.tick_params(labelsize=labelsize)
#    axarr[i].set_xlabel(r'$\theta_x\ $'+'[arcmin]', fontsize = fontsize-5)
#    axarr[i].set_ylabel(r'$\theta_y\ $'+'[arcmin]', fontsize = fontsize-5)
#    axarr[i].tick_params(labelsize = labelsize)
#    axarr[i].set_title(titstr, fontsize = fontsize-5)
#fig.tight_layout()
#plt.show()

#fig.savefig(save_loc_results+'/dipoles_check.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  



#%%
print('Checking likelhihood curves ...')
L_arr_sorted = np.load(save_loc_results+'/likelihoods.npy', allow_pickle=True)
L_ip_arr_sorted = np.load(save_loc_results+'/likelihoods_finer.npy', allow_pickle=True)
nber_runs = 10
s,e = 0,10
L_arr_sorted = [L_arr_sorted[0][s:e], L_arr_sorted[1][s:e], L_arr_sorted[2][s:e], 
                L_arr_sorted[3][s:e], L_arr_sorted[4][s:e], L_arr_sorted[5][s:e]]

L_ip_arr_sorted = [L_ip_arr_sorted[0][s:e], L_ip_arr_sorted[1][s:e], L_ip_arr_sorted[2][s:e],
                   L_ip_arr_sorted[3][s:e], L_ip_arr_sorted[4][s:e], L_ip_arr_sorted[5][s:e]]
mass_int_finer = np.arange(min(mass_int),max(mass_int),0.001) 
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(L_arr_sorted)):   
    if i in [0,1,2]:
        x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int, L_arr_sorted[i], normalize = True, finer_reso = False)
        print(median_value_comb, error_comb) 
        comb_L_arr.append(L_comb)
        M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int_finer, L_ip_arr_sorted[i], normalize = True, finer_reso = True)
        print(median_value_comb_finer, error_comb_finer) 
        comb_L_ip_arr.append(L_ip_comb)
        comb_median_mass_arr.append(median_value_comb_finer)
        comb_error_arr.append(error_comb_finer)
    else:
        x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int, L_arr_sorted[i], normalize = True, finer_reso = False)
        print(median_value_comb, error_comb) 
        comb_L_arr.append(L_comb)
        M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int, L_arr_sorted[i], normalize = True, finer_reso = True)
        print(median_value_comb_finer, error_comb_finer) 
        comb_L_ip_arr.append(L_ip_comb)
        comb_median_mass_arr.append(median_value_comb_finer)
        comb_error_arr.append(error_comb_finer)
combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
    
    
file = open(save_loc_results+'/mass_results.txt',"w")
file.write(combined_median_masses_results_file)
file.close()
#comb_L_arr = np.load(save_loc_results+'/comb_likelihoods.npy', allow_pickle=True)
#comb_L_ip_arr = np.load(save_loc_results+'/comb_likelihoods_finer.npy', allow_pickle=True)
nber_runs_check = nber_runs
start = 0
end = 10000
start_finer = 0
end_finer = 10000

fig, axarr = plt.subplots(len(L_arr_sorted), 1, figsize=(10,18))
titles = [str(noiseval_white_arr[0]), str(noiseval_white_arr[1]), str(noiseval_white_arr[2]), 
          str(noiseval_white_arr[3]), str(noiseval_white_arr[4]), str(noiseval_white_arr[5])]


for i in range(len(axarr)):
    titstr = titles[i]
    axarr[i].set_title(titstr, fontsize = fontsize)
    for j in  range(nber_runs_check):
        axarr[i].plot(mass_int, L_arr_sorted[i][j], ls = '-', marker = 'x', color = 'blue')
        axarr[i].plot(mass_int_finer, L_ip_arr_sorted[i][j], ls = '-', marker = '', color = 'deepskyblue')   
    axarr[i].plot(mass_int, comb_L_arr[i], ls = '-', marker = 'x', color = 'red')
    axarr[i].plot(mass_int_finer[:990], comb_L_ip_arr[i][:990], ls = '-', marker = '', color = 'firebrick')
    axarr[i].axvline(M_input*1e-14, color = 'forestgreen', ls = '--')
    axarr[i].set_xlabel(r'$M_{200c} \ [10^{14}M_\odot]$', fontsize = fontsize-6)
    axarr[i].set_ylabel('Normalized '+r'$\mathcal{L}$', fontsize = fontsize-6)
    axarr[i].tick_params(labelsize = labelsize-6)
plt.show()
fig.savefig(save_loc_results+'/likelihoods_check.eps', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)    

#%%

median_error_arr_sorted = np.load(save_loc_results+'/error_results.npy')
nber_runs = 10
s,e = 0,10
median_error_arr_sorted = [median_error_arr_sorted[0][s:e], median_error_arr_sorted[1][s:e], median_error_arr_sorted[2][s:e],
                           median_error_arr_sorted[3][s:e], median_error_arr_sorted[4][s:e], median_error_arr_sorted[5][s:e]]

frac_mass_unc_arr = np.asarray(np.mean(median_error_arr_sorted, axis = 1))/(M_input*1e-14) #/np.asarray(comb_median_mass_arr)
frac_err_unc_arr = np.asarray(np.std(median_error_arr_sorted, axis = 1))/(M_input*1e-14) #/np.asarray(comb_median_mass_arr)

my_arr = np.asarray(frac_mass_unc_arr)
my_err = np.asarray(frac_err_unc_arr)
print(my_arr)
print(my_err)
scal_fac = np.sqrt(nber_clus/(nber_runs*nber_clus))
for i in range(len(my_arr)):
    my_arr[i] = np.round(my_arr[i], 8)*scal_fac
    my_err[i] = np.round(my_err[i], 8)*scal_fac
print(my_arr)
print(my_err)  
np.save(save_loc_results+'/fractional_mass_uncertainties.npy', my_arr)
np.save(save_loc_results+'/fractional_error_uncertainties.npy', my_err)

scal_fac = np.sqrt(100000./(nber_runs*nber_clus))
qe_arr =  np.asarray([0.0175118, 0.02292831, 0.03323149, 0.05606527, 0.07843721, 0.08927969])*1e-1*scal_fac
mle_arr = np.asarray([0.0095446, 0.01742045, 0.0251993, 0.04917473, 0.06890442, 0.0864266])*1e-1*scal_fac
fig, ax = plt.subplots( figsize=(6, 4))
ax.set_xscale('log')
ax.set_yscale('log')

ax.errorbar(noiseval_white_arr, my_arr, my_err, color = color_arr[0], ls = '', marker = 'o', markersize = 6.5, elinewidth = 2.5, label = 'This Work')
ax.plot(noiseval_white_arr, qe_arr,  color = color_arr[1], ls = '', marker = '^', markersize = 6.5, label = 'QE')
ax.plot(noiseval_white_arr, mle_arr,  color = color_arr[2], ls = '', marker = 'd',markersize = 6.5, label = 'MLE')

ax.set_xlabel(r'$\Delta T_{\rm white}\ [\mu \rm K- \rm arcmin]$', fontsize = fontsize)
ax.set_ylabel(r'$\Delta M/M_{\rm input}$', fontsize = fontsize)
ax.tick_params(direction='in', axis='both', length=0, width=0, which = 'major', top = 'on', bottom = 'on', left = 'on', right = 'on', labelsize = labelsize)
ax.tick_params(direction='in', axis='both', length=0, width=0, which = 'minor', top = 'on', bottom = 'on', left = 'on', right = 'on', labelsize = labelsize)
ax.legend(ncol = 1, prop={'size': labelsize-2}, loc = 'upper left') 
ax.set_ylim(0.0005, 0.015)
ax.set_xlim(0.05, 10)
ax.grid(which = 'both', axis = 'both', lw = 0.5, alpha = 0.25, color = 'black')
plt.show()
fig.savefig(save_loc_results+'/estimator_comparison.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)



