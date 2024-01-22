"""
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import mockobs
import os
if not os.path.exists("sims/lensed_mockobs"):
    os.makedirs("sims/lensed_mockobs")
if not os.path.exists("sims/unlensed_mockobs"):
    os.makedirs("sims/unlensed_mockobs")
    
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import matplotlib.pyplot as plt
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import lensing_estimator 
import stats
import os
if not os.path.exists("models"):
    os.makedirs("models")


nber_clus = 2500
nber_rand = 50000
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
mass = 2e14
mass_int = np.arange(0, 4, 0.01)


maps_rand1 = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg1, cutouts_aligned_rand1, weights_gradient_rand1 = lensing_estimator.get_aligned_stack(maps_rand1, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                    cutout_size_for_grad_est_am = 6, 
                                                                                    l_cut = 2000)   
del maps_rand1


maps_rand2 = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg2, cutouts_aligned_rand2, weights_gradient_rand2 = lensing_estimator.get_aligned_stack(maps_rand2, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                    cutout_size_for_grad_est_am = 6, 
                                                                                    l_cut = 2000)   
del maps_rand2


maps_rand3 = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg3, cutouts_aligned_rand3, weights_gradient_rand3 = lensing_estimator.get_aligned_stack(maps_rand3, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                    cutout_size_for_grad_est_am = 6, 
                                                                                    l_cut = 2000)   
del maps_rand3


maps_clus = np.load('sims/lensed_mockobs/60arcmin_2e14Msolar_1.0arcmin_2.0uKarcmin_2500sims.npy')
stack_aligned_clus, cutouts_aligned_clus, weights_gradient_clus = lensing_estimator.get_aligned_stack(maps_clus, map_params, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                  cutout_size_for_grad_est_am = 6, 
                                                                                  l_cut = 2000)   
del maps_clus


models = np.load('models/60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus.npy')


plt.imshow(models[200])
plt.colorbar()
plt.show()
plt.subplots()
 
    

stack_dipole1 = stack_aligned_clus - stack_bg1
cutouts_dipole1 = []
for i in range(len(cutouts_aligned_clus)):
    cutouts_dipole1.append(cutouts_aligned_clus[i]-stack_bg1)


stack_dipole2 = stack_aligned_clus - stack_bg2
cutouts_dipole2 = []
for i in range(len(cutouts_aligned_clus)):
    cutouts_dipole2.append(cutouts_aligned_clus[i]-stack_bg2)
    

stack_dipole3 = stack_aligned_clus - stack_bg3
cutouts_dipole3 = []
for i in range(len(cutouts_aligned_clus)):
    cutouts_dipole3.append(cutouts_aligned_clus[i]-stack_bg3)


plt.imshow(stack_bg1-stack_bg2)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_bg1-stack_bg3)
plt.colorbar()
plt.show()


covariance_matrix1, correlation_matrix1 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole1, 
                                                                                            weights_gradient_clus, 
                                                                                            map_params, 
                                                                                            cutout_size_am = 10, 
                                                                                            cutout_size_for_grad_est_am = 6, 
                                                                                            l_cut = 2000, l = l, cl = cl, 
                                                                                            cl_noise = cl_noise, 
                                                                                            correct_for_tsz = False)

covariance_matrix2, correlation_matrix2 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole2, 
                                                                                            weights_gradient_clus, 
                                                                                            map_params, 
                                                                                            cutout_size_am = 10, 
                                                                                            cutout_size_for_grad_est_am = 6, 
                                                                                            l_cut = 2000, l = l, cl = cl, 
                                                                                            cl_noise = cl_noise, 
                                                                                            correct_for_tsz = False)

covariance_matrix3, correlation_matrix3 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole3, 
                                                                                            weights_gradient_clus, 
                                                                                            map_params, 
                                                                                            cutout_size_am = 10, 
                                                                                            cutout_size_for_grad_est_am = 6, 
                                                                                            l_cut = 2000, l = l, cl = cl, 
                                                                                            cl_noise = cl_noise, 
                                                                                            correct_for_tsz = False)


plt.imshow(covariance_matrix1-covariance_matrix2)
plt.colorbar()
plt.show()
plt.imshow(covariance_matrix1-covariance_matrix3)
plt.colorbar()
plt.show()


x1, L1, median_value1, error1 = stats.run_ml(stack_dipole1, models, covariance_matrix1, mass_int)
print(median_value1, error1) 
M_ip1, L_ip1 = stats.likelihood_finer_reso(x1, L1, delta=0.001)
L_ip1 /= max(L_ip1) 
median_value_finer1, error_finer1 = stats.get_width_from_sampling(M_ip1, L_ip1)
print(median_value_finer1, error_finer1) 
plt.plot(x1, L1, label = 'model '+str(i), ls = '', marker = 'x')
plt.plot(M_ip1, L_ip1, label = 'finer model '+str(i), ls = '', marker = '.')
plt.subplots()

x2, L2, median_value2, error2 = stats.run_ml(stack_dipole2, models, covariance_matrix2, mass_int)
print(median_value2, error2) 
M_ip2, L_ip2 = stats.likelihood_finer_reso(x2, L2, delta=0.001)
L_ip2 /= max(L_ip2) 
median_value_finer2, error_finer2 = stats.get_width_from_sampling(M_ip2, L_ip2)
print(median_value_finer2, error_finer2) 
plt.plot(x2, L2, label = 'model '+str(i), ls = '', marker = 'x')
plt.plot(M_ip2, L_ip2, label = 'finer model '+str(i), ls = '', marker = '.')
plt.subplots()

x3, L3, median_value3, error3 = stats.run_ml(stack_dipole3, models, covariance_matrix3, mass_int)
print(median_value3, error3) 
M_ip3, L_ip3 = stats.likelihood_finer_reso(x3, L3, delta=0.001)
L_ip3 /= max(L_ip3) 
median_value_finer3, error_finer3 = stats.get_width_from_sampling(M_ip3, L_ip3)
print(median_value_finer3, error_finer3) 
plt.plot(x3, L3, label = 'model '+str(i), ls = '', marker = 'x')
plt.plot(M_ip3, L_ip3, label = 'finer model '+str(i), ls = '', marker = '.')
"""


import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import mockobs
import os
if not os.path.exists("sims/lensed_mockobs"):
    os.makedirs("sims/lensed_mockobs")
if not os.path.exists("sims/unlensed_mockobs"):
    os.makedirs("sims/unlensed_mockobs")
    
import matplotlib.pyplot as plt
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import lensing_estimator 
import stats
import os
if not os.path.exists("models"):
    os.makedirs("models")


nber_clus = 2500
nber_rand = 50000
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
mass = 2e14
mass_int = np.arange(1.45, 2.5, 0.05)
mass_int[0] = 0 
print(mass_int)



#models2 = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl,
#                                  cl_noise = cl_noise, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, 
#                                  l_cut = 2000, average = 20)                                  
models2 = np.load('models/60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version23.npy')
#models2 = np.load('models/60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version23.npy')


#models3 = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, 
#                                  cl_noise = cl_noise, cutout_size_am = 10, cutout_size_for_grad_est_am = 6, 
#                                  l_cut = 2000, average = 20)
#np.save('models/60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version33.npy', models3)
models3 = np.load('models/60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version33.npy')




plt.imshow(models2[20]-models3[20])
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(models2[20]-models3[20])
plt.colorbar()
plt.show()
plt.subplots()
   

    
maps_rand = np.load('sims/unlensed_mockobs/60arcmin_1.0arcmin_2.0uKarcmin_50000sims.npy')
stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                    cutout_size_for_grad_est_am = 6, 
                                                                                    l_cut = 2000)   
del maps_rand


x1_arr, L1_arr = [], []
M_ip1_arr, L_ip1_arr = [], []
x2_arr, L2_arr = [], []
M_ip2_arr, L_ip2_arr = [], []
x3_arr, L3_arr = [], []
M_ip3_arr, L_ip3_arr = [], []

for i in range(5):
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

    covariance_matrix, correlation_matrix = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole, 
                                                                                                weights_gradient_clus, 
                                                                                                map_params, 
                                                                                                cutout_size_am = 10, 
                                                                                                cutout_size_for_grad_est_am = 6, 
                                                                                                l_cut = 2000, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False)

    mass_int = np.arange(0, 4, 0.01)
    x1, L1, median_value1, error1 = stats.run_ml(stack_dipole, models1, covariance_matrix, mass_int, finer_reso = False)
    x1_arr.append(x1)
    L1_arr.append(L1)
    print(median_value1, error1) 
    M_ip1, L_ip1, median_value_finer1, error_finer1 = stats.run_ml(stack_dipole, models1, covariance_matrix, mass_int, finer_reso = True)
    M_ip1_arr.append(M_ip1)
    L_ip1_arr.append(L_ip1)
    print(median_value_finer1, error_finer1) 

    mass_int = np.arange(1.45, 2.5, 0.05)
    mass_int[0] = 0 
    x2, L2, median_value2, error2 = stats.run_ml(stack_dipole, models2, covariance_matrix, mass_int, finer_reso = False)
    x2_arr.append(x2)
    L2_arr.append(L2)
    print(median_value2, error2) 
    M_ip2, L_ip2, median_value_finer2, error_finer2 = stats.run_ml(stack_dipole, models2, covariance_matrix, mass_int, finer_reso = True)
    M_ip2_arr.append(M_ip2)
    L_ip2_arr.append(L_ip2)
    print(median_value_finer2, error_finer2) 

    mass_int = np.arange(0.9, 3.0, 0.1)
    mass_int[0] = 0 
    x3, L3, median_value3, error3 = stats.run_ml(stack_dipole, models3, covariance_matrix, mass_int, finer_reso = False)
    x3_arr.append(x3)
    L3_arr.append(L3)
    print(median_value3, error3) 
    M_ip3, L_ip3, median_value_finer3, error_finer3 = stats.run_ml(stack_dipole, models3, covariance_matrix, mass_int, finer_reso = True)
    M_ip3_arr.append(M_ip3)
    L_ip3_arr.append(L_ip3)
    print(median_value_finer3, error_finer3) 


x1_comb, L1_comb, median_value_comb1, error_comb1 = stats.combined_likelihood(x1, L1_arr, normalize = True, finer_reso = False)
print(median_value_comb1, error_comb1) 
M_ip1_comb, L_ip1_comb, median_value_comb1, error_comb1 = stats.combined_likelihood(x1, L1_arr, normalize = True, finer_reso = True)
print(median_value_comb1, error_comb1) 
x2_comb, L2_comb, median_value_comb2, error_comb2 = stats.combined_likelihood(x2, L2_arr, normalize = True, finer_reso = False)
print(median_value_comb2, error_comb2) 
M_ip2_comb, L_ip2_comb, median_value_comb2, error_comb2 = stats.combined_likelihood(x2, L2_arr, normalize = True, finer_reso = True)
print(median_value_comb2, error_comb2) 
x3_comb, L3_comb, median_value_comb3, error_comb3 = stats.combined_likelihood(x3, L3_arr, normalize = True, finer_reso = False)
print(median_value_comb3, error_comb3) 
M_ip3_comb, L_ip3_comb, median_value_comb3, error_comb3 = stats.combined_likelihood(x3, L3_arr, normalize = True, finer_reso = True)
print(median_value_comb3, error_comb3) 


for i in range(5):
    plt.plot(x1_arr[i], L1_arr[i], ls = '-', marker = '', color = 'blue')
    plt.plot(x1_comb, L1_comb, ls = '', marker = 'o', color = 'green')
    plt.plot(M_ip1_arr[i], L_ip1_arr[i], ls = '-', marker = '', color = 'orange')
    plt.plot(M_ip1_comb, L_ip1_comb, ls = '--', color = 'red')
    
plt.subplots()  
for i in range(5):    
    plt.plot(x2_arr[i], L2_arr[i], ls = '', marker = 'x', color = 'blue')
    plt.plot(x2_comb, L2_comb, ls = '', marker = 'o', color = 'green')
    plt.plot(M_ip2_arr[i], L_ip2_arr[i], ls = '-', marker = '', color = 'orange')
    plt.plot(M_ip2_comb, L_ip2_comb, ls = '--', color = 'red')
    
plt.subplots()
for i in range(5):  
    plt.plot(x3_arr[i], L3_arr[i], ls = '', marker = 'x', color = 'blue')
    plt.plot(x3_comb, L3_comb, ls = '', marker = 'o', color = 'green')
    plt.plot(M_ip3_arr[i], L_ip3_arr[i], ls = '-', marker = '', color = 'orange')
    plt.plot(M_ip3_comb, L_ip3_comb, ls = '--', color = 'red')


"""
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import mockobs    
import matplotlib.pyplot as plt
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import lensing_estimator 
import stats
import os


nber_clus = 2500
nber_rand = 50000
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
mass = 1.47e14


fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/ksz_tsz/mdpl2_cutouts_for_tszksz_clus_detection_M8.5e+13to1.15e+14_z0.67to0.75_32995haloes_boxsize121.0am_dx0.5am_beam1.0am.npz'
cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
mass_z_key = list(cutouts_dic.keys())[0]
cutouts = cutouts_dic[mass_z_key]
#scale_fac = fg.compton_y_to_delta_Tcmb(freq = 150, uK = True)
tsz_cutouts = []
keys = []
for kcntr, keyname in enumerate( cutouts ):
    keys.append(keyname[-1])
    tsz_cutout = cutouts[keyname]['y']#*scale_fac
    tsz_cutouts.append(tsz_cutout[61:181, 61:181])

#%%
plt.imshow(np.mean(tsz_cutouts[:2500], axis = 0)[50:70, 50:70])
plt.colorbar()
plt.subplots()

maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [mass, cosmo.concentration_parameter(mass, z, 0.674), z], bl = bl, nl = nl, nber_obs = nber_clus)

maps_clus_tsz  = np.copy(maps_clus)

import random
for j in range(nber_clus):           
    rand_sel = random.randint(0, len(tsz_cutouts)-1)         
    maps_clus_tsz[j] = maps_clus_tsz[j] + tsz_cutouts[rand_sel]  

stack_aligned_clus_large, cutouts_aligned_clus_large, weights_gradient_clus_large = lensing_estimator.get_aligned_stack(maps_clus_tsz, map_params, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, cutout_size_am = 60, 
                                                                                  cutout_size_for_grad_est_am = 6, 
                                                                                  l_cut = 2000)   


stack_aligned_clus_small, cutouts_aligned_clus_small, weights_gradient_clus_small = lensing_estimator.get_aligned_stack(maps_clus_tsz, map_params, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                  cutout_size_for_grad_est_am = 6, 
                                                                                  l_cut = 2000)   

#%%
maps_rand = mockobs.cmb_mock_data(map_params, l, cl, cluster = [mass, cosmo.concentration_parameter(mass, z, 0.674), z], bl = bl, nl = nl, nber_obs = 20000)
stack_bg, _, _ = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, cutout_size_am = 10, 
                                                                                  cutout_size_for_grad_est_am = 6, 
                                                                                  l_cut = 2000)   
plt.imshow(stack_aligned_clus_large[50:70, 50:70]-stack_bg)
plt.colorbar()
plt.subplots()
plt.imshow(stack_aligned_clus_small-stack_bg)
plt.colorbar()
plt.subplots()
plt.imshow(stack_aligned_clus_large[50:70, 50:70]-stack_aligned_clus_small)
plt.colorbar()
"""
