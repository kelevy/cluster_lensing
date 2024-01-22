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
color_arr = ['darkorchid', 'lightseagreen', 'goldenrod', 'crimson', 'silver']
import cosmo
import lensing
import lensing_estimator
import mockobs
import stats
import utils
import ilc


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
beam_fwhm = 1.4 # arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
experiments = ['spt3g', 'so', 'so_fyst', 's4wide']
z = 0.7
mass_reso = 0.01
mass_min = 0.0
mass_max = 6.0
mass_int = np.arange(mass_min, mass_max, mass_reso)
M_input = 1.47e14
use_unlensed = False

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 19
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
    
save_loc_results = '/data/gpfs/projects/punim1922/cluster_lensing/results/forecasts'
if not os.path.exists(save_loc_results):
    os.makedirs(save_loc_results)


####################################################################################################################################
#%%

"""
fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tszksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am_spt3g.npz'
cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()

mass_z_key = list(cutouts_dic.keys())[0]
cutouts = cutouts_dic[mass_z_key]
#scale_fac = fg.compton_y_to_delta_Tcmb(freq = 150, uK = True)
tsz_cutouts = []
for kcntr, keyname in enumerate( cutouts ):
    tsz_cutout = cutouts[keyname]['y']#*scale_fac
    tsz_cutouts.append(tsz_cutout[61:181, 61:181])

tsz_stack = np.mean(tsz_cutouts[:2500], axis = 0)[54:66, 54:66]
_, extent = utils.make_grid([tsz_stack.shape[0], reso_arcmin, tsz_stack.shape[1], reso_arcmin])
fig, ax = plt.subplots(figsize=(5,5))
im = ax.imshow(tsz_stack, extent = extent, cmap = cm.RdBu_r)
ax.axvline(0, color = 'forestgreen')
ax.axhline(0, color = 'forestgreen')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb = fig.colorbar(im, cax = cax)
title = 'tSZ Stack'
divider = make_axes_locatable(ax) 
plt.show() 
np.save('/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am_spt3g.npy', tsz_cutouts)
"""

#%%


print('Getting kSZ maps ...')
fname = '/data/gpfs/projects/punim1922/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
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
fig.savefig('/data/gpfs/projects/punim1922/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_2500stack_boxsize60.0am_dx0.5am.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
"""
print('Getting tSZ maps ...')
tsz_maps = []
for exp in experiments:
    fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am_'+exp+'.npy'
    tsz_cutouts = np.load(fname, allow_pickle=True)
    tsz_maps.append(tsz_cutouts)

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
    fig.savefig('//Volumes/Extreme_SSD/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am_'+exp+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
    
    del tsz_cutouts
"""    

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
 
#%%    
print('Generating ILC residuals for each experiment ...')


for exp in experiments:
    l, cl_residual, res_ilc_dic, weights = ilc.residuals_and_weights(components = 'all', experiment = exp, cov_from_sims = True)
    np.save(save_loc_results+'/residuals_'+exp+'.npy', cl_residual)
    np.save(save_loc_results+'/res_ilc_'+exp+'.npy', res_ilc_dic)
    np.save(save_loc_results+'/weights_ilc_'+exp+'.npy', weights)


cl_noise_arr = []
for exp in experiments:
    cl_noise = np.load(save_loc_results+'/residuals_'+exp+'.npy', allow_pickle=True)
    cl_noise_arr.append(cl_noise)
   

fig, ax = plt.subplots(figsize=(9,6))

ax.set_yscale('log')
ax.plot(l, cl, label = 'CMB-TT', color = color_arr[-1], ls = '-')
label_arr = ['SPT-3G', 'SO', 'SO + FYST', 'S4-Wide']
ls_arr = ['--', '-.', ':', '-']
for i, exp in enumerate(experiments):
    _, _, res_ilc_dic, _ = ilc.residuals_and_weights(components = 'all', experiment = exp, cov_from_sims = True)
    ax.plot(l, cl_noise_arr[i]-res_ilc_dic['ksz'], label = label_arr[i], color = color_arr[i], ls = ls_arr[i])
ax.set_ylabel('Residual Power '+r'$ \ C_\ell \ [\mu$'+'K'+r'$^2]$', fontsize = fontsize)
ax.set_xlabel('Multipole '+r' \ $\ell$', fontsize = fontsize)
ax.tick_params(labelsize = 20)
ax.tick_params(direction='in', axis='both', length=6, width=1, which = 'major', top = 'on', bottom = 'on', left = 'on', right = 'on', labelsize = labelsize)
ax.tick_params(direction='in', axis='both', length=3, width=1, which = 'minor', top = 'on', bottom = 'on', left = 'on', right = 'on', labelsize = labelsize)
ax.set_xlim(100, 10000)
ax.set_ylim(2e-6, 1e-3)
ax.legend(loc = 1, ncol = 1, prop={'size': labelsize-2}) 
plt.show()

fig.savefig(save_loc_results+'/residuals.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


print('Generating extragalactic residual noise maps ...')
for j in range(len(experiments)):
    print(experiments[j])
    print('for background ...')
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy'):
        sims_extragal_noise = []
        for i in tqdm(range(nber_rand)):         
            sim_extragal_noise = utils.make_gaussian_realization(l, cl_noise_arr[j], map_params)
            sims_extragal_noise.append(sim_extragal_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy', sims_extragal_noise)
        del sims_extragal_noise
    print('for lensed mock obs ...') 
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy'):
        sims_extragal_noise = []
        for i in tqdm(range(nber_runs*nber_clus)):            
            sim_extragal_noise = utils.make_gaussian_realization(l, cl_noise_arr[j], map_params)
            sims_extragal_noise.append(sim_extragal_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy', sims_extragal_noise)
        del sims_extragal_noise
    print('for models ...')
    if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy'):
        sims_extragal_noise = []
        for i in tqdm(range(average*nber_clus)):            
            sim_extragal_noise = utils.make_gaussian_realization(l, cl_noise_arr[j], map_params)
            sims_extragal_noise.append(sim_extragal_noise)
        np.save(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy', sims_extragal_noise)
        del sims_extragal_noise 
        
        
###################################################################################################################################
#%%


print('Generating unlensed mock obs  ...')
for j in range(len(experiments)):
    print(experiments[j])
    if not os.path.exists(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[j]+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy'):
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy')
        sims_extragal_noise = np.load(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy')
        mockobs_unlensed = []
        for i in tqdm(range(nber_rand)): 
            mockob_unlensed = utils.convolve(sims_cmb_unlensed[i] + sims_extragal_noise[i], l, np.sqrt(bl), mapparams = map_params) 
            mockobs_unlensed.append(mockob_unlensed)
        np.save(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[j]+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy', mockobs_unlensed) 
        
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
        fig.savefig(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[j]+'uKarcmin_unlensed_mockob.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
    
        del sims_cmb_unlensed
        del sims_extragal_noise
        del mockobs_unlensed
        
        
print('Generating lensed mock obs  ...')
randoms = np.load(save_loc_mockobs_lensed+'/randoms.npy') 
for j in range(len(experiments)):
    print(experiments[j])
    if not os.path.exists(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+experiments[j]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy'):
        sims_cmb_lensed = np.load(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy') 
        sims_extragal_noise = np.load(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
        fname = '/data/gpfs/projects/punim1922/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am_'+experiments[j]+'.npy'
        tsz_cutouts = np.load(fname, allow_pickle=True)
        mockobs_lensed = []
        for i in tqdm(range(nber_runs*nber_clus)):
            mockob_lensed = utils.convolve(sims_cmb_lensed[i] +  ksz_cutouts[randoms[i]] + tsz_cutouts[randoms[i]] + sims_extragal_noise[i], l, np.sqrt(bl), mapparams = map_params) 
            mockobs_lensed.append(mockob_lensed)
        np.save(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+experiments[j]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', mockobs_lensed) 
        del sims_cmb_lensed
        del sims_extragal_noise
        del tsz_cutouts
        del mockobs_lensed  
   
        
###################################################################################################################################
#%%


# generating models
print('Generating models ...')
models_arr = []
for i in range(len(experiments)):
    
    print(experiments[i])
    if not os.path.exists(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy'):
        print('Generating current models ...')
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy')
        sims_extragal_noise = np.load(save_loc_sims_noise+'/60arcmin_'+experiments[j]+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy')
        models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = None,
                                            cl_noise = cl_noise_arr[i], cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                            l_cut = l_cut, maps_cmb_unlensed = sims_cmb_unlensed, maps_noise = None, maps_extragal_noise = sims_extragal_noise, average = average,
                                            apply_noise = False, use_unlensed = use_unlensed)
        np.save(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy', models)
        del sims_cmb_unlensed
        del sims_extragal_noise
        
    else:
        print('Loading current models ...')
        models = np.load(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy')
    
    models_arr.append(models)  
    

#%%
for i in range(len(experiments)):
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
    
    fig.savefig(save_loc_models+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
    

###################################################################################################################################
#%%


# generating background stack
print('Generating background stack ...')
for i in range(len(experiments)):
    print(experiments[i])
    if not os.path.exists(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_rand)+'sims.npy'):
        mockobs_unlensed = np.load(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy') 
        stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(mockobs_unlensed, map_params, l = l, cl = cl, 
                                                                                                    cl_noise = cl_noise_arr[i], cutout_size_am = cutout_size_am, 
                                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                    l_cut = l_cut)  
        del mockobs_unlensed
        np.save(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
    else:
        stack_bg = np.load(save_loc_results+'/background_stack_cutoutsize'+str(cutout_size_am)+'arcmin_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(nber_rand)+'sims.npy') 

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
    
    fig.savefig(save_loc_results+'/background_stack_noiseval'+experiments[i]+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  

###################################################################################################################################
#%%


print('Starting analysis ...')


x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
median_mass_arr = []
pvalue_arr = []

maps = []
for i in range(len(experiments)):
    print('experiment: ', experiments[i])
    mockobs_lensed_baseline = np.load(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_'+str(beam_fwhm)+'arcmin_'+experiments[i]+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', allow_pickle = True) 
    maps.append(mockobs_lensed_baseline) 

stacks_dipole_all_arr = []
covariance_matrices_all_arr = []


for i in tqdm(range(nber_runs)):    
    x_arr_curr = []
    L_arr_curr = []
    M_ip_arr_curr = []
    L_ip_arr_curr = []
    median_mass_arr_curr = []    
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
                                                        
    print('Computing tSZ stacks ...')
    stacks_tsz = []
    for j in range(len(maps)):                                                                                                                                                                               
        stack_tsz, _ = lensing_estimator.get_tsz(maps = maps[j][i*nber_clus:i*nber_clus+nber_clus], 
                                                    map_params = map_params, 
                                                    weights_gradient = weights_gradient_arr[j], 
                                                    cutout_size_am = cutout_size_am)
        stacks_tsz.append(stack_tsz)

                                      
    print('Computing dipole stacks ...')
    stacks_dipole = []
    for j in range(len(maps)):
        stack_dipole_corrected = stacks_aligned[j] - stack_bg - stacks_tsz[j] 
        stacks_dipole.append(stack_dipole_corrected)
    stacks_dipole_all_arr.append(stacks_dipole)   
   
    
    print('Computing covariance matrices ...')
    covariance_matrix_arr, correlation_matrix_arr = [], []
    for j in range(len(maps)):  
        covariance_matrix_corrected, correlation_matrix_corrected = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned_arr[j], 
                                                                                                                        weights_gradient_arr[j], 
                                                                                                                        map_params, 
                                                                                                                        cutout_size_am = cutout_size_am, 
                                                                                                                        cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                        l_cut = l_cut, l = l, cl = cl, 
                                                                                                                        cl_noise = cl_noise_arr[j], 
                                                                                                                        stack_bg = stack_bg,
                                                                                                                        stack_tsz = stacks_tsz[j])
                                                                                                           
                                                                                                                
                                                                                                                
        covariance_matrix_arr.append(covariance_matrix_corrected)
        correlation_matrix_arr.append(correlation_matrix_corrected)

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
    pvalue_arr.append(pvalue_arr_curr)

np.save(save_loc_results+'/all_stacked_dipoles.npy', stacks_dipole_all_arr)
np.save(save_loc_results+'/all_covariance_matrices.npy', covariance_matrices_all_arr)
np.save(save_loc_results+'/pvalues.npy', pvalue_arr)


#%%

print('Checking dipoles ...')

_, extent = utils.make_grid([stacks_dipole[0].shape[0], reso_arcmin, stacks_dipole[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(stacks_dipole), 1, figsize=(20,20))


for i in range(len(axarr)):
    titstr = experiments[i]
    im = axarr[i].imshow(stacks_dipole[i], extent = extent, cmap = cm.RdBu_r)
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

fig.savefig(save_loc_results+'/dipoles_check.eps', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  
 
#%%
print('Sorting individual runs after case ...')
L_arr_sorted = []
L_ip_arr_sorted = []
median_mass_arr_sorted = []
for i in range(len(stacks_dipole)):
    L_arr_sorted_curr = []
    L_ip_arr_sorted_curr = []
    median_mass_arr_sorted_curr = []
    for j in range(nber_runs):
        L_arr_sorted_curr.append(L_arr[j][i])
        L_ip_arr_sorted_curr.append(L_ip_arr[j][i]) 
        median_mass_arr_sorted_curr.append(median_mass_arr[j][i]) 
    L_arr_sorted.append(L_arr_sorted_curr)
    L_ip_arr_sorted.append(L_ip_arr_sorted_curr)
    median_mass_arr_sorted.append(median_mass_arr_sorted_curr)
    
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
np.save(save_loc_results+'/comb_median_masses.npy', comb_median_mass_arr)
np.save(save_loc_results+'/comb_mass_errors.npy', comb_error_arr)


combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
    
    
file = open(save_loc_results+'/mass_results.txt',"w")
file.write(combined_median_masses_results_file)
file.close()


def signal_to_noise(lnpdf):
    delta_chisq = 2*(max(lnpdf) - lnpdf[0])
    snr = np.sqrt(delta_chisq)
    return snr

snr_spt_3g = signal_to_noise(comb_L_ip_arr[0])
snr_so = signal_to_noise(comb_L_ip_arr[1])
snr_fyst = signal_to_noise(comb_L_ip_arr[2])
snr_s4wide = signal_to_noise(comb_L_ip_arr[3])

snr_arr = [snr_spt_3g, snr_so, snr_fyst, snr_s4wide]
np.save(save_loc_results+'/snr.npy', snr_arr) 

snr_results = "{0:11.3f}".format(snr_spt_3g) + "\n"
snr_results = "{0:11.3f}".format(snr_so) + "\n"
snr_results += "{0:11.3f}".format(snr_fyst) + "\n"          
snr_results += "{0:11.3f}".format(snr_s4wide) + "\n"

file = open(save_loc_results+'/snr.txt',"w")
file.write(snr_results)
file.close() 


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
L_arr_sorted = [L_arr_sorted[0][s:e], L_arr_sorted[1][s:e], L_arr_sorted[2][s:e], L_arr_sorted[3][s:e]]
L_ip_arr_sorted = [L_ip_arr_sorted[0][s:e], L_ip_arr_sorted[1][s:e], L_ip_arr_sorted[2][s:e], L_ip_arr_sorted[3][s:e]]
mass_int_finer = np.arange(min(mass_int),max(mass_int),0.001) 
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(L_arr_sorted)):   
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


def signal_to_noise(lnpdf):
    delta_chisq = 2*(max(lnpdf) - lnpdf[0])
    snr = np.sqrt(delta_chisq)
    return snr

snr_spt_3g = signal_to_noise(np.log(comb_L_ip_arr[0]))
snr_so = signal_to_noise(np.log(comb_L_ip_arr[1]))
snr_fyst = signal_to_noise(np.log(comb_L_ip_arr[2]))
snr_s4wide = signal_to_noise(np.log(comb_L_ip_arr[3]))

snr_arr = [snr_spt_3g, snr_so, snr_fyst, snr_s4wide]
np.save(save_loc_results+'/snr.npy', snr_arr) 

snr_results = "{0:11.3f}".format(snr_spt_3g) + "\n"
snr_results = "{0:11.3f}".format(snr_so) + "\n"
snr_results += "{0:11.3f}".format(snr_fyst) + "\n"          
snr_results += "{0:11.3f}".format(snr_s4wide) + "\n"

file = open(save_loc_results+'/snr.txt',"w")
file.write(snr_results)
file.close() 

nber_runs_check = nber_runs
start = 0
end = 10000
start_finer = 0
end_finer = 10000

fig, axarr = plt.subplots(len(L_arr_sorted), 1, figsize=(10,18))

for i in range(len(axarr)):
    titstr = experiments[i]
    axarr[i].set_title(titstr, fontsize = fontsize-6)
    for j in  range(nber_runs_check):
        axarr[i].plot(mass_int[start:end], L_arr_sorted[i][j][start:end], ls = '-', marker = 'x', color = 'blue')
        axarr[i].plot(mass_int_finer[start_finer:end_finer], L_ip_arr_sorted[i][j][start_finer:end_finer], ls = '-', marker = '', color = 'deepskyblue')   
    axarr[i].plot(mass_int[start:end], comb_L_arr[i][start:end], ls = '-', marker = 'x', color = 'red')
    axarr[i].plot(mass_int_finer[start_finer:end_finer], comb_L_ip_arr[i][start_finer:end_finer], ls = '-', marker = '', color = 'firebrick')
    axarr[i].axvline(M_input*1e-14, color = 'forestgreen', ls = '--')
   # axarr[i].set_xlabel(r'$M_{200c}\ $'+'[10M_\sun]', fontsize = fontsize-6)
   # axarr[i].set_ylabel(r'$\text{Normalized}\ $'+'\mathcal{L}', fontsize = fontsize-6)
    axarr[i].tick_params(labelsize = labelsize-6)
plt.show()
fig.savefig(save_loc_results+'/likelihoods_check.eps', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)    


#%%

snr_arr = np.load(save_loc_results+'/snr.npy') 
scale_fracs = [np.sqrt(7000/(nber_runs*nber_clus)), np.sqrt(25000/(nber_runs*nber_clus)), np.sqrt(25000/(nber_runs*nber_clus)), np.sqrt(100000/(nber_runs*nber_clus))]
snr_arr_scaled = [snr_arr[i]*scale_fracs[i] for i in range(len(snr_arr))]
print(snr_arr_scaled)
print(100./np.asarray(snr_arr_scaled))
snr_dic={}

#experiments - 
cases_arr=['']
snr_dic['spt3g']=[snr_arr_scaled[0]]
snr_dic['so']=[snr_arr_scaled[1]] #list contains SNR for baseline, with centroid shifts, with kSZ, with tSZ correction, all
snr_dic['so_fyst']=[snr_arr_scaled[2]] #list contains SNR for baseline, with centroid shifts, with kSZ, with tSZ correction, all
snr_dic['s4wide']=[snr_arr_scaled[3]] #list contains SNR for baseline, with centroid shifts, with kSZ, with tSZ correction, all
exp_dic = {'spt3g': 'SPT-3G', 'so': 'SO', 'so_fyst': 'SO + FYST', 's4wide': 'S4-Wide'}
nber_dic = {'spt3g': r'$N_{\rm clus}:$' '7,000', 'so': r'$N_{\rm clus}:$' '25,000', 
            'so_fyst': r'$N_{\rm clus}:$' '25,000', 's4wide': r'$N_{\rm clus}:$' '100,000'}

#make bar plots
fsval = 12
yscale_val = None #'log'
ymin, ymax = 0., 80.
exparr = snr_dic.keys()
xcntr = 0.
barwidth = 1.
exp_gap = barwidth * 2.
xincrement = exp_gap*barwidth
hatch_arr = ['///', '\\\\\\', 'xxx', 'o0o']
#ax = subplot(111, yscale = yscale_val)
fig, ax = plt.subplots(figsize=(8.6,4.8))
for expcntr, expname in enumerate( exparr ): #loop for experiments
    snr_arr = snr_dic[expname]
    ax.bar(xcntr, snr_arr[0], width = barwidth+0.6, color = color_arr[expcntr], 
           edgecolor = color_arr[expcntr], fill = False, hatch=hatch_arr[expcntr])
    xcntr += barwidth
    xcntr += xincrement - 1
    ax.axvline(xcntr-1, color = 'gray') 
ax.tick_params(direction='in', axis='x', length=0, width=0, which = 'major', top = 'on', bottom = 'on', left = 'on', right = 'on', labelsize = labelsize)
ax.set_ylim(ymin, ymax)
ax.set_xlim(-1, xcntr - xincrement + 1)

plt.setp(ax.get_xticklabels(which = 'both'), visible=False)
yloc = [ymin-5.2, ymin-5.2, ymin-5.2, ymin-5.2]
xcntr = [-0.5, 1.82, 3.35, 5.5]
for expcntr, expname in enumerate( exparr ):
    ax.text(xcntr[expcntr], yloc[expcntr], exp_dic[expname], fontsize = fontsize-4)
    
ax.text(2.2, ymin-11.5, 'Experiment', fontsize = fontsize)

yloc = [ymin+20, ymin+29, ymin+31, ymin+64]
xcntr = [-0.75, 1.18, 3.18, 5.115]
for expcntr, expname in enumerate( exparr ):
    ax.text(xcntr[expcntr], yloc[expcntr], nber_dic[expname], fontsize = fontsize-4)


ax.set_ylabel('Lensing SNR', fontsize = fontsize)
for label in ax.get_xticklabels(): label.set_fontsize(fontsize)
for label in ax.get_yticklabels(): label.set_fontsize(fontsize)

plt.savefig(save_loc_results+'/forecasts.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)
