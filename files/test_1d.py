import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os
if not os.path.exists("/Volumes//Extreme_SSD/cluster_lensing/models"):
    os.makedirs("/Volumes//Extreme_SSD/cluster_lensing/models")
if not os.path.exists("/Volumes//Extreme_SSD/cluster_lensing/sims/unlensed_mockobs"):
    os.makedirs("/Volumes//Extreme_SSD/cluster_lensing/sims/unlensed_mockobs")
if not os.path.exists("/Volumes//Extreme_SSD/cluster_lensing/results"):
    os.makedirs("/Volumes//Extreme_SSD/cluster_lensing/results")
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import lensing_estimator
import mockobs
import stats
import utils


####################################################################################################################################


nber_clus = 2500
nber_rand = 50000
cutout_size_am = 10
cutout_size_for_grad_est_am = 6
l_cut = 2000
average = 20
nber_runs = 1
map_params = [120, 0.5, 120, 0.5] # nx, dx, ny, dy
beam_fwhm = 1.0 # arcmin
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
z = 0.7
mass_reso = 0.01
mass_min = 0.0
mass_max = 3.0
mass_int = np.arange(mass_min, mass_max, mass_reso)
M_input = 1.47e14
nx = 120
ny = 120
dx = 0.5
dy = 0.5


###################################################################################################################################
#%%

# Generating models
print('Generating models ...')
if not os.path.exists('/Volumes//Extreme_SSD/cluster_lensing/models/10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy'):
    models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, 
                                  cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                  l_cut = l_cut, average = average)
    np.save('/Volumes//Extreme_SSD/cluster_lensing/models/10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy', models)
else:
    models = np.load('/Volumes//Extreme_SSD/cluster_lensing/models/10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy')
    
    
models_1d = []
for model in models:
    models_1d.append(np.mean(model, axis = 0))
    
fig = plt.figure()
for model_1d in models_1d:
    plt.plot(model_1d)
plt.axhline(0)
plt.show()


###################################################################################################################################
#%%

# Generating background stack
print('Generating background stack ...')
if not os.path.exists('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy'):
    maps_rand = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
    stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                    l_cut = l_cut)  
    del maps_rand
    np.save('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
else:
    stack_bg = np.load('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy') 


###################################################################################################################################
#%%

x_arr_baseline = []
L_arr_baseline = []
M_ip_arr_baseline = []
L_ip_arr_baseline = []
mass_arr_baseline = []

x_arr_ksz = []
L_arr_ksz = []
M_ip_arr_ksz = []
L_ip_arr_ksz = []
mass_arr_ksz = []



for i in tqdm(range(nber_runs)):    

    maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [M_input, cosmo.concentration_parameter(M_input, z, 0.674), z], nber_obs = nber_clus, lensing_reso = 1)
    maps_clus_arr_baseline = np.copy(maps_clus)
        
    
    for j in range(nber_clus):
        white_noise_map = utils.make_gaussian_realization(l, nl, map_params) 
        maps_clus_arr_baseline[j] = utils.convolve(maps_clus_arr_baseline[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map


    stack_aligned_baseline, cutouts_aligned_baseline, weights_gradient_baseline = lensing_estimator.get_aligned_stack(maps_clus_arr_baseline, 
                                                                                                                       map_params, 
                                                                                                                       l = l, cl = cl, 
                                                                                                                       cl_noise = cl_noise, 
                                                                                                                       cutout_size_am = cutout_size_am, 
                                                                                                                       cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                       l_cut = l_cut)      

                                                                     

    stack_dipole_baseline = stack_aligned_baseline - stack_bg
    stack_dipole_ksz = np.mean(stack_aligned_baseline - stack_bg, axis = 0)

        
    plt.imshow(stack_dipole_baseline)
    plt.colorbar()
    plt.show()
    plt.subplots()
    plt.plot(stack_dipole_ksz)
    plt.show()

    covariance_matrix_baseline, correlation_matrix_baseline = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned_baseline, 
                                                                                                                  weights_gradient_baseline, 
                                                                                                                  map_params, 
                                                                                                                  cutout_size_am = cutout_size_am, 
                                                                                                                  cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                  l_cut = l_cut, l = l, cl = cl, 
                                                                                                                  cl_noise = cl_noise, 
                                                                                                                  correct_for_tsz = False, 
                                                                                                                  stack_bg = stack_bg)  
    
    covariance_matrix_ksz, correlation_matrix_ksz = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned_baseline, 
                                                                                                weights_gradient_baseline, 
                                                                                                map_params, 
                                                                                                cutout_size_am = cutout_size_am, 
                                                                                                cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                l_cut = l_cut, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False,
                                                                                                stack_bg = stack_bg,
                                                                                                onedim = True)
    

  
    print('Run', i+1)
    x_baseline, L_baseline, median_value_baseline, error_baseline = stats.run_ml(stack_dipole_baseline, models, covariance_matrix_baseline, mass_int, finer_reso = False)
    x_arr_baseline.append(x_baseline)
    L_arr_baseline.append(L_baseline)
    print(median_value_baseline, error_baseline) 
    M_ip_baseline, L_ip_baseline, median_value_finer_baseline, error_finer_baseline = stats.run_ml(stack_dipole_baseline, models, covariance_matrix_baseline, mass_int, finer_reso = True)
    M_ip_arr_baseline.append(M_ip_baseline)
    L_ip_arr_baseline.append(L_ip_baseline)
    print(median_value_finer_baseline, error_finer_baseline) 
    mass_arr_baseline.append(median_value_finer_baseline)
   
    
    x_ksz, L_ksz, median_value_ksz, error_ksz = stats.run_ml(stack_dipole_ksz, models_1d, covariance_matrix_ksz, mass_int, finer_reso = False)
    x_arr_ksz.append(x_ksz)
    L_arr_ksz.append(L_ksz)
    print(median_value_ksz, error_ksz) 
    M_ip_ksz, L_ip_ksz, median_value_finer_ksz, error_finer_ksz= stats.run_ml(stack_dipole_ksz, models_1d, covariance_matrix_ksz, mass_int, finer_reso = True)
    M_ip_arr_ksz.append(M_ip_ksz)
    L_ip_arr_ksz.append(L_ip_ksz)
    print(median_value_finer_ksz, error_finer_ksz) 
    mass_arr_ksz.append(median_value_finer_ksz)
         

likelihood_arr = [L_ip_arr_baseline, L_ip_arr_ksz]
mass_results_arr = [mass_arr_baseline, mass_arr_ksz]

             
print('\n')
print('Combined Results:')    
x_comb_baseline, L_comb_baseline, median_value_comb_baseline, error_comb_baseline = stats.combined_likelihood(x_baseline, L_arr_baseline, normalize = True, finer_reso = False)
print(median_value_comb_baseline, error_comb_baseline) 
M_ip_comb_baseline, L_ip_comb_baseline, median_value_comb_finer_baseline, error_comb_finer_baseline = stats.combined_likelihood(x_baseline, L_arr_baseline, normalize = True, finer_reso = True)
print(median_value_comb_finer_baseline, error_comb_finer_baseline) 

x_comb_ksz, L_comb_ksz, median_value_comb_ksz, error_comb_ksz = stats.combined_likelihood(x_ksz, L_arr_ksz, normalize = True, finer_reso = False)
print(median_value_comb_ksz, error_comb_ksz) 
M_ip_comb_ksz, L_ip_comb_ksz, median_value_comb_finer_ksz, error_comb_finer_ksz = stats.combined_likelihood(x_ksz, L_arr_ksz, normalize = True, finer_reso = True)
print(median_value_comb_finer_ksz, error_comb_finer_ksz) 



comb_likelihood_arr = [L_ip_comb_baseline, L_ip_comb_ksz]
comb_mass_arr = [median_value_comb_finer_baseline, median_value_comb_finer_ksz]
comb_error_arr = [error_comb_finer_baseline, error_comb_finer_ksz]




###################################################################################################################################
#%%


start = 1000
end = 2000
for i in range(nber_runs):   
    plt.plot(x_arr_baseline[i][100:200], L_arr_baseline[i][100:200], ls = '-', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_baseline[i][start:end], L_ip_arr_baseline[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
#plt.plot(x_comb_baseline[100:200], L_comb_baseline[100:200], ls = '-', marker = 'o', color = 'orange')
#plt.plot(M_ip_comb_baseline[start:end], L_ip_comb_baseline[start:end], ls = '-', marker = '', color = 'red')  
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in range(nber_runs):   
    plt.plot(x_arr_ksz[i][100:200], L_arr_ksz[i][100:200], ls = '-', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_ksz[i][start:end], L_ip_arr_ksz[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
#plt.plot(x_comb_ksz[100:200], L_comb_ksz[100:200], ls = '-', marker = 'o', color = 'orange')
#plt.plot(M_ip_comb_ksz[start:end], L_ip_comb_ksz[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()



#%%


mass_results_arr = [mass_results_arr[0], mass_results_arr[1]]
comb_mass_arr = [comb_mass_arr[0], comb_mass_arr[1]]
comb_error_arr = [comb_error_arr[0], comb_error_arr[1]]

fig, ax = plt.subplots(figsize=(8,6))
xvals = [0.06, 0.06]
yvals = [0.15, 0.75]
titles_arr = ['2D','1d']
ax.set_ylim(0.,1)
markers = ['o', '^']
ax.axvline(0, color = 'black', ls = '-')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize = 22)
ax.axes.yaxis.set_ticks([])


ax.set_xlabel(r'$M_{\rm fit}-M_{\rm true}$', fontsize = 28)
for i in range(len(xvals)):
    ax.text(xvals[i], yvals[i], titles_arr[i], fontsize = 22)
    
#for i in range(len(mass_results_arr)):
    #yval = [yvals[i] for j in range(len(mass_results_arr[i]))]
    #ax.plot(np.asarray(mass_results_arr[i])-1.47, yval, color = color_arr[i], marker = markers[i], markersize = 12, ls = '', alpha = 0.25)
for i in range(len(comb_mass_arr)):  
    ax.errorbar((np.round(comb_mass_arr[i], 3)-1.47), yvals[i], xerr = np.round(comb_error_arr[i], 3), elinewidth = 4, color = color_arr[i],  marker = markers[i], markersize = 12, ls = '', capsize=5, capthick=3)



