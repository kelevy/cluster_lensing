import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import lensing_estimator 
import mockobs
import stats
import os
if not os.path.exists("models"):
    os.makedirs("models")

nber_clus = 2500
nber_rand= 50000
nx, ny = 120, 120
dx, dy = 0.5, 0.5
map_params = [nx, dx, ny, dy] 
beam_fwhm = 1.0 # arcmin
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
z = 0.7
mass_int = np.arange(1.475, 2.5, 0.025)
mass_int[0] = 0
print(mass_int)


#models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, 
#                                  cl_noise = cl_noise, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, 
#                                  l_cut = 2000, average = 20)
models = np.load('models/60arcmin_massint1.5to2.5_massres0.025_2.0uKarcmin_1.0arcmin_2500clus_avg20.npy')


plt.imshow(models[20])
plt.colorbar()
plt.show()
plt.subplots()
 

#maps_rand = np.load('sims/unlensed_mockobs/60arcmin_1.0arcmin_2.0uKarcmin_50000sims.npy')
maps_rand = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                    cutout_size_for_grad_est_am = 6, 
                                                                                    l_cut = 2000)  
#np.save('sims/unlensed_mockobs/stack_10arcmin_1.0arcmin_2.0uKarcmin_50000sims.npy', stack_bg) 
del maps_rand

mass = 2e14
x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
for i in range(20):
    maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [mass, cosmo.concentration_parameter(mass, z, 0.674), z], bl = bl, nl = nl, nber_obs = nber_clus)
    stack_aligned_clus, cutouts_aligned_clus, weights_gradient_clus = lensing_estimator.get_aligned_stack(maps_clus, map_params, l = l, cl = cl, 
                                                                                      cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                      cutout_size_for_grad_est_am = 6, 
                                                                                      l_cut = 2000)   
    del maps_clus
        
    
    stack_dipole = stack_aligned_clus - stack_bg
    cutouts_dipole = []
    for i in range(len(cutouts_aligned_clus)):
        cutouts_dipole.append(cutouts_aligned_clus[i]-stack_bg)
    
    
    plt.imshow(stack_bg)
    plt.colorbar()
    plt.show()
    plt.subplots()
    plt.imshow(stack_dipole)
    plt.colorbar()
    plt.show()
    
    
    covariance_matrix, correlation_matrix = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole, 
                                                                                                weights_gradient_clus, 
                                                                                                map_params, 
                                                                                                cutout_size_am = 10, 
                                                                                                cutout_size_for_grad_est_am = 6, 
                                                                                                l_cut = 2000, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False)
    plt.imshow(covariance_matrix)
    plt.colorbar()
    plt.show()
    
    
    x, L, median_value, error = stats.run_ml(stack_dipole, models, covariance_matrix, mass_int, finer_reso = False)
    x_arr.append(x)
    L_arr.append(L)
    print(median_value, error) 
    M_ip, L_ip, median_value_finer, error_finer = stats.run_ml(stack_dipole, models, covariance_matrix, mass_int, finer_reso = True)
    M_ip_arr.append(M_ip)
    L_ip_arr.append(L_ip)
    print(median_value_finer, error_finer) 

#%%       
x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(x, L_arr, normalize = True, finer_reso = False)
print(median_value_comb, error_comb) 
M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(x, L_arr, normalize = True, finer_reso = True)
print(median_value_comb_finer, error_comb_finer) 

for i in range(20):
   # plt.plot(x_arr[i], L_arr[i], ls = '-', marker = '', color = 'blue')
    plt.plot(x_comb, L_comb, ls = '', marker = 'o', color = 'green')
   # plt.plot(M_ip_arr[i], L_ip_arr[i], ls = '-', marker = '', color = 'orange')
    plt.plot(M_ip_comb, L_ip_comb, ls = '--', color = 'red')   
    
#%%
0.05/np.sqrt(20)