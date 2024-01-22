#%%
print('Importing relevant modules ...')
# general modules
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
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

# own modules
import cosmo
import lensing
import mockobs
import lensing_estimator
import stats
import utils
import foregrounds as fg


####################################################################################################################################
#%%


print('Defining relevant parameters ...')
# Try using 25000 cutouts for models, should this not give better results? Why?
# Would averaging over 10 models give better results? Why? 
# Are first and second approach equivalent? Why?
# nber_fit = 25000
version = 2
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
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = np.load('/data/gpfs/projects/punim1922/cluster_lensing/cmb/planck2018_TTTEEE_lowE_lensing_lensedCls.npy')
l, cl_extra = fg.extragalactic_power_spectrum(freq=150, components = 'all')
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = cl_extra + mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)

z = 0.7
mass_reso = 0.01
mass_min_arr = [0.0, 0.0, 0.0, 0.0]
mass_max_arr = [6.0, 6.0, 6.0, 6.0] 
mass_int_arr = [np.arange(mass_min_arr[0], mass_max_arr[0], mass_reso), 
                np.arange(mass_min_arr[1], mass_max_arr[1], mass_reso), 
                np.arange(mass_min_arr[2], mass_max_arr[2], mass_reso),
                np.arange(mass_min_arr[3], mass_max_arr[3], mass_reso),]
M_input_arr = [1e14, 2e14, 4e14, 8e14]
use_unlensed = False

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 34
labelsize = 30

# for saving
save_loc_sims_unlensed = '/data/gpfs/projects/punim1922/cluster_lensing/sims/unlensed_cmb/'
#'/Users/kevinlevy/Documents/cluster_lensing/sims/unlensed_cmb/'
if not os.path.exists(save_loc_sims_unlensed):
    os.makedirs(save_loc_sims_unlensed)
save_loc_sims_lensed = '/data/gpfs/projects/punim1922/cluster_lensing/sims/lensed_cmb/'
#'/Users/kevinlevy/Documents/cluster_lensing/sims/lensed_cmb/'
if not os.path.exists(save_loc_sims_lensed):
    os.makedirs(save_loc_sims_lensed)
save_loc_sims_noise = '/data/gpfs/projects/punim1922/cluster_lensing/sims/noise_maps/'
#'/Users/kevinlevy/Documents/cluster_lensing/sims/noise_maps/'
if not os.path.exists(save_loc_sims_noise):
    os.makedirs(save_loc_sims_noise)

save_loc_mockobs_unlensed = '/data/gpfs/projects/punim1922/cluster_lensing/mockobs/unlensed/'
#'/Users/kevinlevy/Documents/cluster_lensing/mockobs/unlensed/'
if not os.path.exists(save_loc_mockobs_unlensed):
    os.makedirs(save_loc_mockobs_unlensed)
save_loc_mockobs_lensed = '/data/gpfs/projects/punim1922/cluster_lensing/mockobs/lensed/'
#'/Users/kevinlevy/Documents/cluster_lensing/mockobs/lensed/'
if not os.path.exists(save_loc_mockobs_lensed):
    os.makedirs(save_loc_mockobs_lensed)

save_loc_models = '/data/gpfs/projects/punim1922/cluster_lensing/models/'
#'/Users/kevinlevy/Documents/cluster_lensing/models/'
if not os.path.exists(save_loc_models):
    os.makedirs(save_loc_models)

save_loc_results = '/data/gpfs/projects/punim1922/cluster_lensing/results/ref1/'
#'/Users/kevinlevy/Documents/cluster_lensing/results/pipeline_validation/'
if not os.path.exists(save_loc_results):
    os.makedirs(save_loc_results)
    
   
###################################################################################################################################
#%%


print('Generating unlensed CMB sims ...')
print('\t for background ...')
if not os.path.exists(save_loc_sims_unlensed+'60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy'):
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = nber_rand)
    np.save(save_loc_sims_unlensed+'60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy', sim_unlensed)
    del sim_unlensed
print('\t for mock data ...')
if not os.path.exists(save_loc_sims_unlensed+'60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy'):
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = nber_runs*nber_clus)
    np.save(save_loc_sims_unlensed+'60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy', sim_unlensed)
    del sim_unlensed
print('\t for models ...')
if not os.path.exists(save_loc_sims_unlensed+'60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy'):
    sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = average*nber_clus)
    np.save(save_loc_sims_unlensed+'60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy', sim_unlensed)
    del sim_unlensed


print('Generating lensed CMB sims ...')
for M_input in M_input_arr:
    print('for Mass ', M_input*1e-14, ' 1e14 ...')
    grid, _ = utils.make_grid(map_params) 
    theta =  np.hypot(grid[0], grid[1]) 
    if not os.path.exists(save_loc_sims_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims_version1.npy'): 
        c200c = cosmo.concentration_parameter(M_input, z, 0.674)
        kappa_map = lensing.NFW_convergence(M_input, c200c, z, 1100, theta, dim = 2)
        alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params)   
        sims_unlensed = np.load(save_loc_sims_unlensed+'60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy')
        sims_lensed1 = []
        sims_lensed2 = []
        for sim_unlensed in tqdm(sims_unlensed):
            extragal_map = utils.make_gaussian_realization(l, cl_extra, map_params)
            sim_lensed1 = lensing.lens_map(sim_unlensed, alpha_vec, map_params)+extragal_map
            sims_lensed1.append(sim_lensed1)
            sim_lensed2 = lensing.lens_map(sim_unlensed+extragal_map, alpha_vec, map_params) 
            sims_lensed2.append(sim_lensed2)
        np.save(save_loc_sims_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(nber_runs*nber_clus)+'lensed_sims_version1.npy', sims_lensed1)  
        np.save(save_loc_sims_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(nber_runs*nber_clus)+'lensed_sims_version2.npy', sims_lensed2)  
        _, extent = utils.make_grid([sims_lensed1[0].shape[0], reso_arcmin, sims_lensed1[0].shape[1], reso_arcmin])
        fig, ax = plt.subplots(figsize=(5,5))
        title = 'Lensed Simulation for Mass ' +str(M_input*1e-14)
        im = ax.imshow(sim_lensed1, extent = extent, cmap = cm.RdBu_r)
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
        fig.savefig(save_loc_sims_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(nber_runs*nber_clus)+'lensed_sims_version1.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
        
        del sims_unlensed
        del sims_lensed1
        del sims_lensed2
        
        
        
print('Generating white noise sims ...')
print('for background ...')
if not os.path.exists(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy'):
    sims_white_noise = []
    for i in tqdm(range(nber_rand)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy', sims_white_noise)
    del sims_white_noise
print('for lensed mock obs ...')
if not os.path.exists(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy'):
    sims_white_noise = []
    for i in tqdm(range(nber_runs*nber_clus)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy', sims_white_noise)
    del sims_white_noise
print('for models ...')
if not os.path.exists(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy'):
    sims_white_noise = []
    for i in tqdm(range(average*nber_clus)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy', sims_white_noise)
    del sims_white_noise 
    
    
###################################################################################################################################
#%%


print('Generating unlensed mock obs  ...')
if not os.path.exists(save_loc_mockobs_unlensed+'60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy'):
    sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy')
    sims_white_noise = np.load(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy')
    mockobs_unlensed = []
    for i in tqdm(range(nber_rand)):
       # sim_cmb_unlensed = #mockobs.cmb_mock_data(map_params, l, cl) #sims_cmb_unlensed[i]
       # sim_white_noise = #utils.make_gaussian_realization(l, nl, map_params) #sims_white_noise[i] 
        mockob_unlensed = utils.convolve(sims_cmb_unlensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]   
        mockobs_unlensed.append(mockob_unlensed)
    np.save(save_loc_mockobs_unlensed+'60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy', mockobs_unlensed) 
    
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
    fig.savefig(save_loc_mockobs_unlensed+'60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_unlensed_mockob.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

    del sims_cmb_unlensed
    del sims_white_noise
    del mockobs_unlensed
    
print('Generating lensed mock obs  ...')
for M_input in M_input_arr:
    print(M_input*1e-14)
    if not os.path.exists(save_loc_mockobs_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs_version'+str(version)+'.npy'):
        sims_cmb_lensed = np.load(save_loc_sims_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims_version'+str(version)+'.npy') 
        sims_white_noise = np.load(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
        mockobs_lensed = []
        for i in tqdm(range(nber_runs*nber_clus)):
            mockob_lensed = utils.convolve(sims_cmb_lensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]
            mockobs_lensed.append(mockob_lensed)
        np.save(save_loc_mockobs_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs_version'+str(version)+'.npy', mockobs_lensed) 
        del sims_cmb_lensed
        del sims_white_noise
        del mockobs_lensed  
    
    
###################################################################################################################################
#%%


print('Generating models ...')
models_arr = []
for i in range(len(mass_int_arr)):
    print(mass_int_arr[i])
    
    if not os.path.exists(save_loc_models+'models_6arcmin_massint'+str(mass_min_arr[i])+'to'+str(mass_max_arr[i])+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy'):
        print('Generating current models ...')
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy')
        sims_white_noise = np.load(save_loc_sims_noise+'60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy')
        models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int_arr[i], z, centroid_shift_value = 0, bl = bl, nl = nl,
                                          cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                          l_cut = l_cut, maps_cmb_unlensed = sims_cmb_unlensed, maps_noise = sims_white_noise , average = average,
                                          use_unlensed = use_unlensed)
        np.save(save_loc_models+'models_6arcmin_massint'+str(mass_min_arr[i])+'to'+str(mass_max_arr[i])+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy', models)
        del sims_cmb_unlensed
        del sims_white_noise
        
    else:
        print('Loading current models ...')
        models = np.load(save_loc_models+'models_6arcmin_massint'+str(mass_min_arr[i])+'to'+str(mass_max_arr[i])+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy')
    
    models_arr.append(models)  
    

#%%

print('Checking models ...')
for i in range(len(mass_int_arr)):
    models_check = [models_arr[i][0], models_arr[i][int(len(mass_int_arr[i])/2)], models_arr[i][-1]]
    _, extent = utils.make_grid([models_check[0].shape[0], reso_arcmin, models_check[0].shape[1], reso_arcmin])
    fig, axarr = plt.subplots(len(models_check), 1, figsize=(10,10))
    dec = len(str(mass_reso).split(".")[1])
    titles = ['Model '+str(np.round(mass_int_arr[i][0], dec)), 'Model '+str(np.round(mass_int_arr[i][int(len(mass_int_arr[i])/2)], dec)), 'Model '+str(np.round(mass_int_arr[i][-1], dec))]
    
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
    
    fig.savefig(save_loc_models+'models_10arcmin_massint'+str(mass_min_arr[i])+'to'+str(mass_max_arr[i])+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

###################################################################################################################################
#%%


# generating background stack
print('Generating background stack ...')
if not os.path.exists(save_loc_results+'background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy'):
    mockobs_unlensed = np.load(save_loc_mockobs_unlensed+'60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy') 
    stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(mockobs_unlensed, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                    l_cut = l_cut)  
    del mockobs_unlensed
    np.save(save_loc_results+'background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
else:
    stack_bg = np.load(save_loc_results+'background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy') 

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

fig.savefig(save_loc_results+'background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

###################################################################################################################################
#%%

print('Getting dipoles and covariance matrices ...')

import inpaint
nx, dx, ny, dy = map_params
x1,x2 = -nx/2. * dx, nx/2. * dx
noofsims = 1000
mask_radius_inner = 4.0 #arcmins
mask_radius_outer = 20.0 #arcmins
noofsims = 1000 
mask_inner = 0 #If 1, the inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
#get ra, dec or map-pixel grid
ra = np.linspace(x1,x2, nx) #arcmins
dec = np.linspace(x1,x2, nx) #arcmins
ra_grid, dec_grid = np.meshgrid(ra,dec)
sigma_dic = inpaint.get_covariance(ra_grid, dec_grid, map_params, l, cl, bl, nl, noofsims, mask_radius_inner, mask_radius_outer)


if not os.path.exists(save_loc_results+'stacked_dipoles_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(M_input_arr[-1]*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims_version'+str(version)+'.npy'):
    stacks_dipole_all = []
    covariance_matrices_all = []
    
       
    for M_input in M_input_arr:
        print('Mass: ', M_input*1e-14)
        maps = np.load(save_loc_mockobs_lensed+'60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs_version'+str(version)+'.npy') 
        stacks_dipole_curr = []
        covariance_matrices_curr = []
        for i in tqdm(range(nber_runs)):    
    
            print('Computing cluster stacks ...')                            
        
            stack_aligned, cutouts_aligned, weights_gradient = lensing_estimator.get_aligned_stack(maps[i*nber_clus:i*nber_clus+nber_clus], 
                                                                                                   map_params, 
                                                                                                   l = l, cl = cl, 
                                                                                                   cl_noise = cl_noise, 
                                                                                                   cutout_size_am = cutout_size_am, 
                                                                                                   cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                   l_cut = l_cut,
                                                                                                   inpaint_map = False, bl = bl, 
                                                                                                   nl = nl, sigma_dic = sigma_dic, 
                                                                                                   mask_radius_inner = mask_radius_inner,
                                                                                                   mask_radius_outer = mask_radius_outer,
                                                                                                   noofsims = noofsims, 
                                                                                                   mask_inner = mask_inner)  
            
            
            print('Computing dipole stacks ..')
            stack_dipole = stack_aligned - stack_bg 
            stacks_dipole_curr.append(stack_dipole)
         
            print('Computing covariance matrices ...')
            covariance_matrix, correlation_matrix = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned, 
                                                                                                        weights_gradient, 
                                                                                                        map_params, 
                                                                                                        cutout_size_am = cutout_size_am, 
                                                                                                        cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                        l_cut = l_cut, l = l, cl = cl, 
                                                                                                        cl_noise = cl_noise, 
                                                                                                        stack_bg = stack_bg,
                                                                                                        stack_tsz = None)
            covariance_matrices_curr.append(covariance_matrix)
          
        stacks_dipole_all.append(stacks_dipole_curr)                                            
        covariance_matrices_all.append(covariance_matrices_curr)
        
    np.save(save_loc_results+'stacked_dipoles_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims_version_'+str(version)+'.npy', stacks_dipole_all)
    np.save(save_loc_results+'covariance_matrices_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims_version'+str(version)+'.npy', covariance_matrices_all)
  
else:
    stacks_dipole_all = np.load(save_loc_results+'stacked_dipoles_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(M_input_arr[-1]*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims_version'+str(version)+'.npy')
    covariance_matrices_all = np.load(save_loc_results+'covariance_matrices_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(M_input_arr[-1]*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims_version'+str(version)+'.npy')
  
    
###################################################################################################################################
#%%

print('Checking dipoles ...')
dipole_checks = [stacks_dipole_all[0][-1], stacks_dipole_all[1][-1], stacks_dipole_all[2][-1], stacks_dipole_all[3][-1],]

_, extent = utils.make_grid([dipole_checks[0].shape[0], reso_arcmin, dipole_checks[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(dipole_checks), 1, figsize=(10,10))
titles = ['1e14', '2e14', '4e14', '8e14']

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

fig.savefig(save_loc_results+'dipoles_check_version'+str(version)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
  
#%%
  
print('Getting likelihoods and mass estimates ...') 
x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
median_mass_arr = []  
pvalue_arr = []

for i in range(len(M_input_arr)):
    print('Mass: ', M_input_arr[i]*1e-14)
    x_arr_curr = []
    L_arr_curr = []
    M_ip_arr_curr = []
    L_ip_arr_curr = []
    median_mass_arr_curr = []  
    pvalue_arr_curr = []
           
    for j in range(nber_runs):  
        x, L, median_value, error = stats.run_ml(stacks_dipole_all[i][j], models_arr[i], covariance_matrices_all[i][j], mass_int_arr[i], finer_reso = False)
        x_arr_curr.append(x)
        L_arr_curr.append(L)
        print(median_value, error) 
        M_ip, L_ip, median_value_finer, error_finer = stats.run_ml(stacks_dipole_all[i][j], models_arr[i], covariance_matrices_all[i][j], mass_int_arr[i], finer_reso = True)
        M_ip_arr_curr.append(M_ip)
        L_ip_arr_curr.append(L_ip)
        median_mass_arr_curr.append(median_value_finer)
        print(median_value_finer, error_finer) 
        
        import scipy as sp
        sp.stats.chisqprob = lambda chisq, df: sp.stats.chi2.sf(chisq, df)
        print('median_value: ', median_value)
        print('mass interval: ', mass_int_arr[i][np.argmax(L)])
        diff_vector = stacks_dipole_all[i][j].flatten() - models_arr[i][np.argmax(L)].flatten()
        cov_inv = sp.linalg.pinv(covariance_matrices_all[i][j])
        chisqval = np.asarray( np.dot(diff_vector.T, np.dot(cov_inv , diff_vector ))).squeeze() 
        pvalue = sp.stats.chisqprob(chisqval,len(diff_vector)-1) 
        pvalue_arr_curr.append(pvalue)
        print('pvalue: ', pvalue)
       
    x_arr.append(x_arr_curr)
    L_arr.append(L_arr_curr)
    M_ip_arr.append(M_ip_arr_curr)
    L_ip_arr.append(L_ip_arr_curr)
    median_mass_arr.append(median_mass_arr_curr)
    pvalue_arr.append(pvalue_arr_curr)
    

print('Getting combined likelihoods and mass estimates ...')  
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(M_input_arr)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    comb_L_arr.append(L_comb)
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
   

np.save(save_loc_results+'likelihoods_version'+str(version)+'.npy', L_arr)
np.save(save_loc_results+'likelihoods_finer_version'+str(version)+'.npy', L_ip_arr)
np.save(save_loc_results+'comb_likelihoods_version'+str(version)+'.npy', comb_L_arr)
np.save(save_loc_results+'comb_likelihoods_finer_version'+str(version)+'.npy', comb_L_ip_arr)
np.save(save_loc_results+'mass_results_version'+str(version)+'.npy', median_mass_arr)
np.save(save_loc_results+'comb_median_masses_version'+str(version)+'.npy', comb_median_mass_arr)
np.save(save_loc_results+'comb_mass_errors_version'+str(version)+'.npy', comb_error_arr)
np.save(save_loc_results+'pvalues_version'+str(version)+'.npy', pvalue_arr)

combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
file = open(save_loc_results+'mass_results_version'+str(version)+'.txt',"w")
file.write(combined_median_masses_results_file)
file.close()


#%%

print('Checking likelhihood curves ...')
L_arr= np.load(save_loc_results+'likelihoods_version'+str(version)+'.npy', allow_pickle=True)
L_ip_arr = np.load(save_loc_results+'likelihoods_finer_version'+str(version)+'.npy', allow_pickle=True)
nber_runs = 10
s,e = 3,13
L_arr = [L_arr[0][s:e], L_arr[1][s:e], L_arr[2][s:e], L_arr[3][s:e]]
L_ip_arr = [L_ip_arr[0][s:e], L_ip_arr[1][s:e], L_ip_arr[2][s:e], L_ip_arr[3][s:e]]
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(M_input_arr)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    comb_L_arr.append(L_comb)
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
file = open(save_loc_results+'mass_results_version'+str(version)+'.txt',"w")
file.write(combined_median_masses_results_file)
file.close()
#comb_L_arr = np.load(save_loc_results+'/comb_likelihoods.npy', allow_pickle=True)
#comb_L_ip_arr = np.load(save_loc_results+'/comb_likelihoods_finer.npy', allow_pickle=True)


nber_runs_check = nber_runs
start_arr = [0, 0, 0, 0]
end_arr = [1000, 1000, 1000, 8000]
start_finer_arr = [0, 0, 0, 0]
end_finer_arr = [1000, 1000, 1000, 8000]
mass_int_finer_arr = [np.arange(min(mass_int_arr[0]),max(mass_int_arr[0]),0.001),
                      np.arange(min(mass_int_arr[1]),max(mass_int_arr[1]),0.001),
                      np.arange(min(mass_int_arr[2]),max(mass_int_arr[2]),0.001),
                      np.arange(min(mass_int_arr[3]),max(mass_int_arr[3]),0.001)]
                    
fig, axarr = plt.subplots(len(L_arr), 1, figsize=(7,15))
titles = ['1e14', '2e14', '414', '8e14']

for i in range(len(axarr)):
    for j in  range(nber_runs_check):
        axarr[i].plot(mass_int_arr[i][start_arr[i]:end_arr[i]], L_arr[i][j][start_arr[i]:end_arr[i]], ls = '-', marker = 'x', color = 'blue')
        axarr[i].plot(mass_int_finer_arr[i][start_finer_arr[i]:end_finer_arr[i]], L_ip_arr[i][j][start_finer_arr[i]:end_finer_arr[i]], ls = '-', marker = '', color = 'palegreen')   
    axarr[i].plot(mass_int_arr[i][start_arr[i]:end_arr[i]], comb_L_arr[i][start_arr[i]:end_arr[i]], ls = '-', marker = 'x', color = 'red')
    axarr[i].plot(mass_int_finer_arr[i][start_finer_arr[i]:end_finer_arr[i]], comb_L_ip_arr[i][start_finer_arr[i]:end_finer_arr[i]], ls = '-', marker = '', color = 'limegreen')
    axarr[i].axvline(M_input_arr[i]*1e-14, color = 'forestgreen', ls = '--')
    axarr[i].set_xlabel(r'$M_{200c} \ [10^{14}M_\odot]$', fontsize = fontsize)
    axarr[i].set_ylabel('Normalized '+r'$\mathcal{L}$', fontsize = fontsize)
    axarr[i].tick_params(labelsize = labelsize-6)
plt.show()
fig.savefig(save_loc_results+'likelihoods_check_version'+str(version)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)    

#%%
print('Plotting likelihood curves ...')
L_ip_arr = np.load(save_loc_results+'likelihoods_finer_version'+str(version)+'.npy', allow_pickle=True)
nber_runs = 10
s,e = 3,13
L_ip_arr = [L_ip_arr[0][s:e], L_ip_arr[1][s:e], L_ip_arr[2][s:e], L_ip_arr[3][s:e]]
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(M_input_arr)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int_arr[i], L_arr[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
file = open(save_loc_results+'mass_results_version'+str(version)+'.txt',"w")
file.write(combined_median_masses_results_file)
file.close()
#comb_L_arr = np.load(save_loc_results+'/comb_likelihoods.npy', allow_pickle=True)
#comb_L_ip_arr = np.load(save_loc_results+'/comb_likelihoods_finer.npy', allow_pickle=True)

start_finer_arr = [0, 0, 0, 0]
end_finer_arr = [1000, 1000, 1000, 8000]
mass_int_finer_arr = [np.arange(min(mass_int_arr[0]),max(mass_int_arr[0]),0.001),
                      np.arange(min(mass_int_arr[1]),max(mass_int_arr[1]),0.001),
                      np.arange(min(mass_int_arr[2]),max(mass_int_arr[2]),0.001),
                      np.arange(min(mass_int_arr[3]),max(mass_int_arr[3]),0.001)]
                    
fig, axarr = plt.subplots(1, len(L_ip_arr), figsize=(23,6))


for i in range(len(axarr)):
    for j in  range(nber_runs):
        axarr[i].plot(mass_int_finer_arr[i][start_finer_arr[i]:end_finer_arr[i]], L_ip_arr[i][j][start_finer_arr[i]:end_finer_arr[i]], ls = '-', marker = '', color = color_arr[0], linewidth=3.0, alpha = 0.25)   
    axarr[i].plot(mass_int_finer_arr[i][start_finer_arr[i]:end_finer_arr[i]], comb_L_ip_arr[i][start_finer_arr[i]:end_finer_arr[i]], ls = '-', marker = '', color = color_arr[1], linewidth=3.0, label = 'Combined')
    axarr[i].axvline(M_input_arr[i]*1e-14, color = color_arr[2], ls = '--', label = 'Input', linewidth=3.0)
    axarr[i].set_xlabel(r'$M_{200c} \ [10^{14}M_\odot]$', fontsize = fontsize)
    if i == 0:
        axarr[i].set_ylabel('Normalized '+r'$\mathcal{L}$', fontsize = fontsize)
    axarr[i].tick_params(labelsize = labelsize)
    axarr[i].legend(loc = 'upper left', prop={'size': labelsize-2})  
    axarr[i].axes.yaxis.set_ticks([])
plt.tight_layout()
plt.show()
fig.savefig(save_loc_results+'pipeline_validation_version'+str(version)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)   

