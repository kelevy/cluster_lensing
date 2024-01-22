#%%
print('Importing relevant modules ...')
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import rcParams
import matplotlib
#from matplotlib import rc;rc('text', usetex=False);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
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
cl = np.load('/data/gpfs/projects/punim1922/cluster_lensing/cmb/planck2018_TTTEEE_lowE_lensing_lensedCls.npy')
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)

z = 0.7
mass_reso = 0.01
mass_min = 0.0
mass_max = 2.5
mass_int = np.arange(mass_min, mass_max, mass_reso)
M_input = 1.47e14
offsets = [0.0, 0.5]
use_unlensed = False
unlensed_analysis = False

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 18
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
    
save_loc_results = '/data/gpfs/projects/punim1922/cluster_lensing/results/ref2_alt'
if not os.path.exists(save_loc_results):
    os.makedirs(save_loc_results)
    
    
###################################################################################################################################
#%%


print('Getting kSZ maps ...')
fname = '/data/gpfs/projects/punim1922/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
ksz_cutouts = np.load(fname)


print('Getting tSZ maps ...')
fname = '/data/gpfs/projects/punim1922/cluster_lensing/sims/extragalactic/mdpl2_cutouts_for_tsz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
tsz_cutouts = np.load(fname)

randoms = np.random.choice(range(0, len(ksz_cutouts)-1), size=nber_runs*nber_clus, replace=True)

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
for offset in offsets:
    if not os.path.exists(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offset)+'_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy'):
        print(offset)
        centroid_shift_value = offset
        sims_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(int(nber_runs*nber_clus))+'unlensed_sims_clus.npy')
        sims_lensed = []
        for i in tqdm(range(nber_runs*nber_clus)):#tqdm(sims_unlensed):   
            x_shift, y_shift = np.random.normal(loc=0.0, scale = centroid_shift_value/(2**0.5)), np.random.normal(loc=0.0, scale= centroid_shift_value/(2**0.5))            
            centroid_shift = [x_shift, y_shift]
            grid, _ = utils.make_grid(map_params, grid_shift = centroid_shift) 
            theta =  np.hypot(grid[0], grid[1]) 
            c200c = cosmo.concentration_parameter(M_input, z, 0.674)
            kappa_map = lensing.NFW_convergence(M_input, c200c, z, 1100, theta, dim = 2)
            alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params)   
            sim_lensed = lensing.lens_map(sims_unlensed[i], alpha_vec, map_params) 
            sims_lensed.append(sim_lensed)
        np.save(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offset)+'_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy', sims_lensed)  
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
print('for models ...')
if not os.path.exists(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy'):
    sims_white_noise = []
    for i in tqdm(range(average*nber_clus)):            
        sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
        sims_white_noise.append(sim_white_noise)
    np.save(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy', sims_white_noise)
    del sims_white_noise 
   
   
###################################################################################################################################
#%%


print('Generating unlensed mock obs  ...')
if not os.path.exists(save_loc_mockobs_unlensed+'/60arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'unlensed_mockobs.npy'):
    sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(nber_rand)+'unlensed_sims_background.npy')
    sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'noise_sims_background.npy')
    mockobs_unlensed = []
    for i in tqdm(range(nber_rand)):
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
for j in range(len(offsets)):
    print(offsets[j])
    if not os.path.exists(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offsets[j])+'_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy'):  
        sims_cmb_lensed = np.load(save_loc_sims_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offsets[j])+'_'+str(int(nber_runs*nber_clus))+'lensed_sims.npy')     
        sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'noise_sims_clus.npy')
        mockobs_lensed = []
        for i in tqdm(range(nber_runs*nber_clus)):   
            mockob_lensed = utils.convolve(sims_cmb_lensed[i] +  ksz_cutouts[randoms[i]] + tsz_cutouts[randoms[i]], l, np.sqrt(bl), mapparams = map_params) + sims_white_noise[i]
            mockobs_lensed.append(mockob_lensed)
        np.save(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offsets[j])+'_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', mockobs_lensed) 
        del sims_cmb_lensed
        del sims_white_noise
        del mockobs_lensed  
        
        
###################################################################################################################################
#%%

models_arr = []
for offset in offsets:
    print(offset)

    if not os.path.exists(save_loc_models+'/models_'+str(cutout_size_am)+'arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_offset'+str(offset)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy'):
        print('Generating current models ...')
        sims_cmb_unlensed = np.load(save_loc_sims_unlensed+'/60arcmin_'+str(int(average*nber_clus))+'unlensed_sims_models_avgrun'+str(average_run)+'.npy')
        sims_white_noise = np.load(save_loc_sims_noise+'/60arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(average*nber_clus))+'noise_sims_models_avgrun'+str(average_run)+'.npy')
           
        models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, 
                                            centroid_shift_value = offset, 
                                            bl = bl, nl = nl,
                                            cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                            cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                            l_cut = l_cut,
                                            maps_cmb_unlensed=sims_cmb_unlensed, 
                                            maps_noise = sims_white_noise,
                                            average = average, 
                                            use_unlensed = use_unlensed)   
        np.save(save_loc_models+'/models_'+str(cutout_size_am)+'arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_offset'+str(offset)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy', models)
        del sims_cmb_unlensed
        del sims_white_noise
    
    else:
        print('Loading current models ...')
        models = np.load(save_loc_models+'/models_'+str(cutout_size_am)+'arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_offset'+str(offset)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'_avgrun'+str(average_run)+'.npy')
    
    models_arr.append(models)  


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
    
fig.savefig(save_loc_results+'/background_stack_noiseval'+str(noiseval_white)+'.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)  


###################################################################################################################################
#%%


print('Starting analysis ...')


x_arr_biased = []
L_arr_biased = []
M_ip_arr_biased = []
L_ip_arr_biased = []
median_mass_arr_biased = []
pvalue_arr_biased = []

x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
median_mass_arr = []
pvalue_arr = []


stacks_dipole_all_arr = []
covariance_matrices_all_arr = []

stacks_dipole_all = []
covariance_matrices_all = []


for j, offset in enumerate(offsets): 

    maps = np.load(save_loc_mockobs_lensed+'/60arcmin_'+str(M_input*1e-14)+'1e14M_sun_offset'+str(offsets[j])+'_kSZtSZ_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(int(nber_runs*nber_clus))+'lensed_mockobs.npy', allow_pickle=True) 
    print(offsets[j])         
    
    stacks_dipole_curr = []
    covariance_matrices_curr = [] 
    
    x_arr_curr_biased = []
    L_arr_curr_biased = []
    M_ip_arr_curr_biased = []
    L_ip_arr_curr_biased = []
    median_mass_arr_curr_biased = []
    pvalue_arr_curr_biased = []

    x_arr_curr = []
    L_arr_curr = []
    M_ip_arr_curr = []
    L_ip_arr_curr = []
    median_mass_arr_curr = []
    pvalue_arr_curr = []

    for i in range(nber_runs):
        stack_aligned, cutouts_aligned, weights_gradient = lensing_estimator.get_aligned_stack(maps[i*nber_clus:i*nber_clus+nber_clus],
                                                                                               map_params, 
                                                                                               l = l, cl = cl, 
                                                                                               cutout_size_am = cutout_size_am, 
                                                                                               cl_noise = cl_noise, 
                                                                                               cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                               l_cut = l_cut)  
                                                                                                                                                                                                         

        
        
        print('Computing dipole stacks ...')  
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
        print('Getting likelihoods and mass estimates ...')
        print('Run', i+1)
        x_biased, L_biased, median_value_biased, error_biased = stats.run_ml(stack_dipole, models_arr[0], covariance_matrix, mass_int, finer_reso = False)
        x_arr_curr_biased.append(x_biased)
        L_arr_curr_biased.append(L_biased)
        plt.plot(x_biased, L_biased)
        print(median_value_biased, error_biased) 
        M_ip_biased, L_ip_biased, median_value_finer_biased, error_finer_biased = stats.run_ml(stack_dipole, models_arr[0], covariance_matrix, mass_int, finer_reso = True)
        M_ip_arr_curr_biased.append(M_ip_biased)
        L_ip_arr_curr_biased.append(L_ip_biased)
        print(median_value_finer_biased, error_finer_biased) 
           
        x, L, median_value, error = stats.run_ml(stack_dipole, models_arr[j], covariance_matrix, mass_int, finer_reso = False)
        x_arr_curr.append(x)
        L_arr_curr.append(L)
        plt.plot(x, L)
        print(median_value, error) 
        M_ip, L_ip, median_value_finer, error_finer = stats.run_ml(stack_dipole, models_arr[j], covariance_matrix, mass_int, finer_reso = True)
        M_ip_arr_curr.append(M_ip)
        L_ip_arr_curr.append(L_ip)
        print(median_value_finer, error_finer) 

        import scipy as sp
        sp.stats.chisqprob = lambda chisq, df: sp.stats.chi2.sf(chisq, df)
        print('median_value: ', median_value_biased)
        print('mass interval: ', mass_int[np.argmax(L_biased)])
        diff_vector_biased = stack_dipole.flatten() - models_arr[0][np.argmax(L_biased)].flatten()
        cov_inv = sp.linalg.pinv(covariance_matrix)
        chisqval_biased = np.asarray( np.dot(diff_vector_biased.T, np.dot(cov_inv , diff_vector_biased))).squeeze() 
        pvalue_biased = sp.stats.chisqprob(chisqval_biased,len(diff_vector_biased)-1) 
        pvalue_arr_curr_biased.append(pvalue_biased)
        print(pvalue_biased)      
        
        print('median_value: ', median_value)
        print('mass interval: ', mass_int[np.argmax(L)])
        diff_vector = stack_dipole.flatten() - models_arr[j][np.argmax(L)].flatten()
        cov_inv = sp.linalg.pinv(covariance_matrix)
        chisqval = np.asarray( np.dot(diff_vector.T, np.dot(cov_inv , diff_vector))).squeeze() 
        pvalue = sp.stats.chisqprob(chisqval,len(diff_vector)-1) 
        pvalue_arr_curr.append(pvalue)
        print(pvalue)
        
    stacks_dipole_all.append(stacks_dipole_curr)                                            
    covariance_matrices_all.append(covariance_matrices_curr)
    
    x_arr_biased.append(x_arr_curr_biased)
    L_arr_biased.append(L_arr_curr_biased)
    M_ip_arr_biased.append(M_ip_arr_curr_biased)
    L_ip_arr_biased.append(L_ip_arr_curr_biased)
    median_mass_arr_biased.append(median_mass_arr_curr_biased)
    pvalue_arr_biased.append(pvalue_arr_curr_biased)
    
    x_arr.append(x_arr_curr)
    L_arr.append(L_arr_curr)
    M_ip_arr.append(M_ip_arr_curr)
    L_ip_arr.append(L_ip_arr_curr)
    median_mass_arr.append(median_mass_arr_curr)
    pvalue_arr.append(pvalue_arr_curr)
    
    del maps    

np.save(save_loc_results+'/stacks_dipole_all.npy', stacks_dipole_all)
np.save(save_loc_results+'/covariance_matrices_all.npy', covariance_matrices_all)
np.save(save_loc_results+'/pvalues_biased.npy', pvalue_arr_biased)
np.save(save_loc_results+'/pvalues.npy', pvalue_arr)


#%%
print('Getting combined likelihoods and mass estimates ...')  
comb_L_arr_biased = []
comb_L_ip_arr_biased  = []
comb_median_mass_arr_biased  = []
comb_error_arr_biased  = []
for i in range(len(offsets)):   
    x_comb_biased, L_comb_biased, median_value_comb_biased, error_comb_biased = stats.combined_likelihood(mass_int, L_arr_biased[i], normalize = True, finer_reso = False)
    print(median_value_comb_biased, error_comb_biased) 
    comb_L_arr_biased.append(L_comb_biased)
    M_ip_comb_biased, L_ip_comb_biased, median_value_comb_finer_biased, error_comb_finer_biased = stats.combined_likelihood(mass_int, L_arr_biased[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer_biased, error_comb_finer_biased) 
    comb_L_ip_arr_biased.append(L_ip_comb_biased)
    comb_median_mass_arr_biased.append(median_value_comb_finer_biased)
    comb_error_arr_biased.append(error_comb_finer_biased)
   

np.save(save_loc_results+'/likelihoods_biased.npy', L_arr_biased)
np.save(save_loc_results+'/likelihoods_finer_biased.npy', L_ip_arr_biased)
np.save(save_loc_results+'/comb_likelihoods_biased.npy', comb_L_arr_biased)
np.save(save_loc_results+'/comb_likelihoods_finer_biased.npy', comb_L_ip_arr_biased)
np.save(save_loc_results+'/mass_results_biased.npy', median_mass_arr_biased)
np.save(save_loc_results+'/comb_median_masses_biased.npy', comb_median_mass_arr_biased)
np.save(save_loc_results+'/comb_mass_errors_biased.npy', comb_error_arr_biased)
np.save(save_loc_results+'/pvalues_biased.npy', pvalue_arr_biased)

combined_median_masses_results_file_biased = ''
for i in range(len(comb_median_mass_arr_biased)):
    combined_median_masses_results_file_biased += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr_biased[i], "+-", comb_error_arr_biased[i]) + "\n"  
file_biased = open(save_loc_results+'/mass_results_biased.txt',"w")
file_biased.write(combined_median_masses_results_file_biased)
file_biased.close()


print('Getting combined likelihoods and mass estimates ...')  
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(offsets)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(mass_int, L_arr[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    comb_L_arr.append(L_comb)
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(mass_int, L_arr[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
   

np.save(save_loc_results+'/likelihoods.npy', L_arr)
np.save(save_loc_results+'/likelihoods_finer.npy', L_ip_arr)
np.save(save_loc_results+'/comb_likelihoods.npy', comb_L_arr)
np.save(save_loc_results+'/comb_likelihoods_finer.npy', comb_L_ip_arr)
np.save(save_loc_results+'/mass_results.npy', median_mass_arr)
np.save(save_loc_results+'/comb_median_masses.npy', comb_median_mass_arr)
np.save(save_loc_results+'/comb_mass_errors.npy', comb_error_arr)
np.save(save_loc_results+'/pvalues.npy', pvalue_arr)

combined_median_masses_results_file = ''
for i in range(len(comb_median_mass_arr)):
    combined_median_masses_results_file += "{0:11.2f}{1:>1}{2:5.2f}".format(comb_median_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  
file = open(save_loc_results+'/mass_results.txt',"w")
file.write(combined_median_masses_results_file)
file.close()



