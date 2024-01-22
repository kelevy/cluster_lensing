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
from matplotlib import rc;rc('text', usetex=False);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
rcParams['font.family'] = 'serif'
rc('text.latex',preamble=r'\usepackage{/Volumes/Extreme_SSD/cluster_lensing/files/apjfonts}')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cosmo
import lensing
import lensing_estimator
import mockobs
import utils


####################################################################################################################################
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
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
z = 0.7
M_input = 1.47e14

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 18
labelsize = 15

# for saving
save_loc_sims_unlensed = '/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_cmb'
if not os.path.exists(save_loc_sims_unlensed):
    os.makedirs(save_loc_sims_unlensed)
save_loc_sims_lensed = '/Volumes/Extreme_SSD/cluster_lensing/sims/lensed_cmb'
if not os.path.exists(save_loc_sims_lensed):
    os.makedirs(save_loc_sims_lensed)
save_loc_sims_noise = '/Volumes/Extreme_SSD/cluster_lensing/sims/noise_maps'
if not os.path.exists(save_loc_sims_noise):
    os.makedirs(save_loc_sims_noise)
    
save_loc_mockobs_unlensed = '/Volumes/Extreme_SSD/cluster_lensing/mockobs/unlensed'
if not os.path.exists(save_loc_mockobs_unlensed):
    os.makedirs(save_loc_mockobs_unlensed)
save_loc_mockobs_lensed = '/Volumes/Extreme_SSD/cluster_lensing/mockobs/lensed'
if not os.path.exists(save_loc_mockobs_lensed):
    os.makedirs(save_loc_mockobs_lensed)
        
save_loc_models = '/Volumes/Extreme_SSD/cluster_lensing/models'
if not os.path.exists(save_loc_models):
    os.makedirs(save_loc_models)
    
save_loc_results = '/Volumes/Extreme_SSD/cluster_lensing/results/lensing_pipeline'
if not os.path.exists(save_loc_results):
    os.makedirs(save_loc_results)
    
   
###################################################################################################################################
#%%


print('Getting kSZ maps ...')
fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
ksz_cutouts = np.load(fname)

ksz_stack = np.mean(ksz_cutouts[:2500], axis = 0)[50:70, 50:70]
_, extent = utils.make_grid([ksz_stack.shape[0], reso_arcmin, ksz_stack.shape[1], reso_arcmin])
fig, ax = plt.subplots(figsize=(5,5))
title = 'kSZ Stack'
im = ax.imshow(ksz_stack, extent = extent, cmap = cm.RdBu_r)
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
fig.savefig('/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_2500stack_boxsize60.0am_dx0.5am.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  


print('Getting tSZ maps ...')
fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
tsz_cutouts = np.load(fname)

tsz_stack = np.mean(tsz_cutouts[:2500], axis = 0)[50:70, 50:70]
_, extent = utils.make_grid([tsz_stack.shape[0], reso_arcmin, tsz_stack.shape[1], reso_arcmin])
fig, ax = plt.subplots(figsize=(5,5))
title = 'tSZ Stack'
im = ax.imshow(tsz_stack, extent = extent, cmap = cm.RdBu_r)
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
fig.savefig('/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_2500stack_boxsize60.0am_dx0.5am.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

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
print('for background ...')
if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy'):
    sims_white_noise = []
    for i in tqdm(range(nber_rand)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy', sims_white_noise)
    del sims_white_noise
print('for lensed mock obs ...') 
if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy'):
    sims_white_noise = []
    for i in tqdm(range(nber_runs*nber_clus)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy', sims_white_noise)
    del sims_white_noise
    
    
###################################################################################################################################
#%%


print('Generating unlensed mock obs  ...')
if not os.path.exists(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy'):
    sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy')
    sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy')
    mockobs_unlensed = []
    for i in tqdm(range(nber_rand)):
       # sim_cmb_unlensed = #mockobs.cmb_mock_data(map_params, l, cl) #sims_cmb_unlensed[i]
       # sim_white_noise = #utils.make_gaussian_realization(l, nl, map_params) #sims_white_noise[i] 
        mockob_unlensed = utils.convolve(sims_cmb_unlensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]   
        mockobs_unlensed.append(mockob_unlensed)
    np.save(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy', mockobs_unlensed) 
    
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
    fig.savefig(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_unlensed_mockob.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

    del sims_cmb_unlensed
    del sims_white_noise
    del mockobs_unlensed
    
print('Generating lensed mock obs  ...')
# for baseline
if not os.path.exists(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy'):
    sims_cmb_lensed = np.load(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy') 
    sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
    mockobs_lensed = []
    for i in tqdm(range(nber_runs*nber_clus)):
        mockob_lensed = utils.convolve(sims_cmb_lensed[i], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]
        mockobs_lensed.append(mockob_lensed)
    np.save(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', mockobs_lensed) 
    del sims_cmb_lensed
    del sims_white_noise
    del mockobs_lensed  
# for kSZ+tSZ
randoms = np.random.choice(range(0, len(tsz_cutouts)-1), size=nber_runs*nber_clus, replace=True)
if not os.path.exists(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy'):
    sims_cmb_lensed = np.load(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy') 
    sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
    mockobs_lensed = []
    for i in tqdm(range(nber_runs*nber_clus)):
        mockob_lensed = utils.convolve(sims_cmb_lensed[i] + ksz_cutouts[randoms[i]] +  tsz_cutouts[randoms[i]], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]
        mockobs_lensed.append(mockob_lensed)
    np.save(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', mockobs_lensed) 
    del sims_cmb_lensed
    del sims_white_noise
    del mockobs_lensed  
    

###################################################################################################################################
#%%


# generating background stack
print('Generating background stack ...')
if not os.path.exists(save_loc_results+'/background_stack_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy'):
    mockobs_unlensed = np.load(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy') 
    stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(mockobs_unlensed, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                    l_cut = l_cut)  
    del mockobs_unlensed
    np.save(save_loc_results+'/background_stack_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
else:
    stack_bg = np.load(save_loc_results+'/background_stack_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy') 


print('Starting analysis ...')
###################################################################################################################################
#%%

mockobs_unlensed = np.load(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy') 


idx = [0,3,38]

plt.subplots()
plt.imshow(mockobs_unlensed[idx[0]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutout1_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_unlensed[idx[1]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutout2_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_unlensed[idx[2]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutout3_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


_, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(mockobs_unlensed[:idx[2]+1], map_params, l = l, cl = cl, 
                                                                                     cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                     cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                     l_cut = l_cut)

plt.subplots()
plt.imshow(cutouts_aligned_rand[idx[0]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutoutaligned1_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_rand[idx[1]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutoutaligned2_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_rand[idx[2]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/randcutoutaligned3_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()

plt.imshow(stack_bg, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/backgroundstack_'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


#%%
del mockobs_unlensed


#%%
mockobs_lensed_baseline = np.load(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy') 


#%%

idx = [0,6,16]

plt.subplots()
plt.imshow(mockobs_lensed_baseline[idx[0]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutout1_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_lensed_baseline[idx[1]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutout2_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_lensed_baseline[idx[2]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutout3_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


stack_aligned_clus, cutouts_aligned_clus, weights_gradient_clus = lensing_estimator.get_aligned_stack(mockobs_lensed_baseline[:2500], 
                                                                                       map_params, 
                                                                                       l = l, cl = cl, 
                                                                                       cl_noise = cl_noise, 
                                                                                       cutout_size_am = cutout_size_am, 
                                                                                       cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                       l_cut = l_cut)  

plt.subplots()
plt.imshow(cutouts_aligned_clus[idx[0]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutoutaligned1_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_clus[idx[1]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutoutaligned2_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_clus[idx[2]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/cluscutoutaligned3_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

plt.subplots()
plt.imshow(stack_aligned_clus, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusterstack_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


#%%
del mockobs_lensed_baseline


#%%

mockobs_lensed_ksztsz = np.load(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy') 


#%%

idx = [0,6,16]

plt.subplots()
plt.imshow(mockobs_lensed_ksztsz[idx[0]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutout1_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_lensed_ksztsz[idx[1]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutout2_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(mockobs_lensed_ksztsz[idx[2]][54:66, 54:66], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutout3_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

#%%
stack_aligned_clus_ksztsz, cutouts_aligned_clus_ksztsz, weights_gradient_clus_ksztsz = lensing_estimator.get_aligned_stack(mockobs_lensed_ksztsz[:2500], 
                                                                                                                  map_params, 
                                                                                                                  l = l, cl = cl, 
                                                                                                                  cl_noise = cl_noise, 
                                                                                                                  cutout_size_am = cutout_size_am, 
                                                                                                                  cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                  l_cut = l_cut)  

plt.subplots()
plt.imshow(cutouts_aligned_clus_ksztsz[idx[0]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutaligned1_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_clus_ksztsz[idx[1]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutaligned2_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_aligned_clus_ksztsz[idx[2]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutaligned3_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

plt.subplots()
plt.imshow(stack_aligned_clus_ksztsz, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusterksztszstack_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

#%%

stack_tsz_mean, stack_tsz_fit, cutouts_rot_tsz = lensing_estimator.get_tsz(mockobs_lensed_ksztsz[:2500], map_params, weights_gradient_clus_ksztsz, cutout_size_am = cutout_size_am, use_gradient_weights = True, 
                                                                           noise_weights_clus = None)

plt.subplots()
plt.imshow(cutouts_rot_tsz[idx[0]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutrand1_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_rot_tsz[idx[1]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutrand2_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(cutouts_rot_tsz[idx[2]], cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/clusksztszcutoutrand3_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

plt.subplots()
plt.imshow(stack_tsz_mean, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/tszstack_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'sims.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


#%%
del mockobs_lensed_ksztsz



#%%

dipole_baseline = stack_aligned_clus-stack_bg
dipole_tsz = stack_aligned_clus_ksztsz-stack_bg
dipole_tsz_corrected = stack_aligned_clus_ksztsz-stack_bg-stack_tsz_mean
plt.subplots()
plt.imshow(dipole_baseline, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/dipole_baseline_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(dipole_tsz, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/dipole_ksztsz_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
plt.subplots()
plt.imshow(dipole_tsz_corrected, cmap = cm.RdBu_r)
plt.axis('off')
plt.savefig(save_loc_results+'/dipole_ksztsz_corrected_'+str(cutout_size_am)+'arcmin_mass'+str(M_input*1e-14)+'1e14Msol'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin.png', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

#%%
plt.show()
