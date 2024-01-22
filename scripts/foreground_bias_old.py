#%%
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
#%%

nber_clus = 2500
nber_rand = 50000
cutout_size_am = 10
cutout_size_for_grad_est_am = 6
l_cut = 2000
average = 1
nber_runs = 20
map_params = [120, 0.5, 120, 0.5] # nx, dx, ny, dy
lensing_reso = 1
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
    models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
                                  cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                  l_cut = l_cut, lensing_reso = lensing_reso, average = average)
    np.save('/Volumes//Extreme_SSD/cluster_lensing/models/10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy', models)
else:
    models = np.load('/Volumes//Extreme_SSD/cluster_lensing/models/10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy')
    
for i in range(len(models)-1):
    plt.imshow(models[i+1]-models[i])
    plt.colorbar()
    plt.title(str(mass_int[i+1])+'-'+str(mass_int[i]))
    plt.show()
fig = plt.figure()
for i in range(len(models)):
    plt.plot(np.mean(models[i], axis = 0))
plt.axhline(0)
plt.show()

###################################################################################################################################
#%%

# Generating background stack
print('Generating background stack ...')
if not os.path.exists('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy'):
    maps_rand = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand, lensing_reso = lensing_reso)
    stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                    cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                    l_cut = l_cut)  
    del maps_rand
    np.save('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy', stack_bg) 
else:
    stack_bg = np.load('/Volumes/Extreme_SSD/cluster_lensing/sims/unlensed_mockobs/stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy') 


###################################################################################################################################


# Getting kSZ and tSZ maps
print('Getting kSZ and tSZ maps ...')
fname = '/Volumes//Extreme_SSD/cluster_lensing/sims/ksz_tsz/mdpl2_cutouts_for_tszksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize121.0am_dx0.5am.npz'
cutouts_dic = np.load(fname, allow_pickle = 1, encoding= 'latin1')['arr_0'].item()
mass_z_key = list(cutouts_dic.keys())[0]
cutouts = cutouts_dic[mass_z_key]
#scale_fac = fg.compton_y_to_delta_Tcmb(freq = 150, uK = True)
tsz_cutouts, ksz_cutouts = [], []
keys = []
for kcntr, keyname in enumerate( cutouts ):
    keys.append(keyname[-1])
    ksz_cutout = cutouts[keyname]['ksz']
    ksz_cutouts.append(ksz_cutout[61:181, 61:181])
    tsz_cutout = cutouts[keyname]['y']#*scale_fac
    tsz_cutouts.append(tsz_cutout[61:181, 61:181])
    
    
#%%
plt.imshow(np.mean(ksz_cutouts[:2500], axis = 0)[50:70, 50:70])
plt.colorbar()
plt.show()
plt.subplots()

plt.imshow(np.mean(tsz_cutouts[:2500], axis = 0)[50:70, 50:70])
plt.colorbar()
plt.show()
plt.subplots()


nx_cutout, ny_cutout = ksz_cutouts[0].shape[0], ksz_cutouts[0].shape[1]
s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)


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

x_arr_tsz = []
L_arr_tsz = []
M_ip_arr_tsz = []
L_ip_arr_tsz = []
mass_arr_tsz = []

x_arr_tsz_corrected1 = []
L_arr_tsz_corrected1 = []
M_ip_arr_tsz_corrected1 = []
L_ip_arr_tsz_corrected1 = []
mass_arr_tsz_corrected1 = []

x_arr_tsz_corrected2 = []
L_arr_tsz_corrected2 = []
M_ip_arr_tsz_corrected2 = []
L_ip_arr_tsz_corrected2 = []
mass_arr_tsz_corrected2 = []

x_arr_ksz_tsz = []
L_arr_ksz_tsz = []
M_ip_arr_ksz_tsz = []
L_ip_arr_ksz_tsz = []
mass_arr_ksz_tsz = []

x_arr_ksz_tsz_corrected1 = []
L_arr_ksz_tsz_corrected1 = []
M_ip_arr_ksz_tsz_corrected1 = []
L_ip_arr_ksz_tsz_corrected1 = []
mass_arr_ksz_tsz_corrected1 = []

x_arr_ksz_tsz_corrected2 = []
L_arr_ksz_tsz_corrected2 = []
M_ip_arr_ksz_tsz_corrected2 = []
L_ip_arr_ksz_tsz_corrected2 = []
mass_arr_ksz_tsz_corrected2 = []


for i in tqdm(range(nber_runs)):    

    maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [M_input, cosmo.concentration_parameter(M_input, z, 0.674), z], nber_obs = nber_clus, lensing_reso = lensing_reso)
    maps_clus_arr_baseline, maps_clus_arr_ksz, maps_clus_arr_tsz  = np.copy(maps_clus), np.copy(maps_clus), np.copy(maps_clus)
    maps_clus_arr_ksz_tsz = np.copy(maps_clus)
        
    for j in range(nber_clus):
        rand_sel = random.randint(0, len(ksz_cutouts)-1)
        maps_clus_arr_ksz[j] = maps_clus_arr_ksz[j] + ksz_cutouts[rand_sel]           
        maps_clus_arr_tsz[j] = maps_clus_arr_tsz[j] + tsz_cutouts[rand_sel]     
        maps_clus_arr_ksz_tsz[j] = maps_clus_arr_ksz_tsz[j] + ksz_cutouts[rand_sel] + tsz_cutouts[rand_sel]  
    
    for j in range(nber_clus):
        white_noise_map = utils.make_gaussian_realization(l, nl, map_params) 
        maps_clus_arr_baseline[j] = utils.convolve(maps_clus_arr_baseline[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map
        maps_clus_arr_ksz[j] = utils.convolve(maps_clus_arr_ksz[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map
        maps_clus_arr_tsz[j] = utils.convolve(maps_clus_arr_tsz[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map
        maps_clus_arr_ksz_tsz[j] = utils.convolve(maps_clus_arr_ksz_tsz[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map
    
    stack_aligned_baseline, cutouts_aligned_baseline, weights_gradient_baseline = lensing_estimator.get_aligned_stack(maps_clus_arr_baseline, 
                                                                                                                       map_params, 
                                                                                                                       l = l, cl = cl, 
                                                                                                                       cl_noise = cl_noise, 
                                                                                                                       cutout_size_am = cutout_size_am, 
                                                                                                                       cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                       l_cut = l_cut)      
    stack_aligned_ksz, cutouts_aligned_ksz, weights_gradient_ksz = lensing_estimator.get_aligned_stack(maps_clus_arr_ksz, 
                                                                                                       map_params, 
                                                                                                       l = l, cl = cl, 
                                                                                                       cl_noise = cl_noise, 
                                                                                                       cutout_size_am = cutout_size_am, 
                                                                                                       cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                       l_cut = l_cut)      
    stack_aligned_tsz, cutouts_aligned_tsz, weights_gradient_tsz = lensing_estimator.get_aligned_stack(maps_clus_arr_tsz, 
                                                                                                       map_params, 
                                                                                                       l = l, cl = cl, 
                                                                                                       cl_noise = cl_noise, 
                                                                                                       cutout_size_am = cutout_size_am, 
                                                                                                       cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                       l_cut = l_cut)      
    stack_tsz, stack_tsz_fit, _ = lensing_estimator.get_tsz(maps_clus_arr_tsz, 
                                                            map_params, 
                                                            weights_gradient_tsz, 
                                                            cutout_size_am = cutout_size_am)
    
   # stack_aligned_ksz_tsz, cutouts_aligned_ksz_tsz, weights_gradient_ksz_tsz = lensing_estimator.get_aligned_stack(maps_clus_arr_ksz_tsz, 
   #                                                                                                                map_params, 
   #                                                                                                                l = l, cl = cl, 
   #                                                                                                                cl_noise = cl_noise, 
   #                                                                                                                cutout_size_am = cutout_size_am, 
   #                                                                                                                cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
   #                                                                                                                l_cut = l_cut)      
   # stack_ksz_tsz, stack_ksz_tsz_fit, _ = lensing_estimator.get_tsz(maps_clus_arr_ksz_tsz, 
   #                                                                 map_params, 
   #                                                                 weights_gradient_ksz_tsz, 
   #                                                                 cutout_size_am = cutout_size_am)
                                                                     

    stack_dipole_baseline = stack_aligned_baseline - stack_bg
    cutouts_dipole_baseline = []
    for k in range(nber_clus):
        cutouts_dipole_baseline.append(cutouts_aligned_baseline[k])
        
    stack_dipole_ksz = stack_aligned_ksz - stack_bg
    cutouts_dipole_ksz = []
    for k in range(nber_clus):
        cutouts_dipole_ksz.append(cutouts_aligned_ksz[k])
        
    stack_dipole_tsz = stack_aligned_tsz - stack_bg
    cutouts_dipole_tsz = []
    for k in range(nber_clus):
        cutouts_dipole_tsz.append(cutouts_aligned_tsz[k])
   
    #stack_dipole_tsz_corrected1 = stack_aligned_tsz - stack_bg - stack_tsz
    #cutouts_dipole_tsz_corrected1 = []
    #for k in range(nber_clus):
    #    cutouts_dipole_tsz_corrected1.append(cutouts_aligned_tsz[k])

   # stack_dipole_tsz_corrected2 = stack_aligned_tsz - stack_bg - stack_tsz_fit
   # cutouts_dipole_tsz_corrected2 = []
   # for k in range(nber_clus):
   #     cutouts_dipole_tsz_corrected2.append(cutouts_aligned_tsz[k])
        
  #  stack_dipole_ksz_tsz = stack_aligned_ksz_tsz - stack_bg
  #  cutouts_dipole_ksz_tsz = []
  #  for k in range(nber_clus):
  #      cutouts_dipole_ksz_tsz.append(cutouts_aligned_ksz_tsz[k])
        
   # stack_dipole_ksz_tsz_corrected1 = stack_aligned_ksz_tsz - stack_bg - stack_ksz_tsz
   # cutouts_dipole_ksz_tsz_corrected1 = []
   # for k in range(nber_clus):
   #     cutouts_dipole_ksz_tsz_corrected1.append(cutouts_aligned_ksz_tsz[k])

   # stack_dipole_ksz_tsz_corrected2 = stack_aligned_ksz_tsz - stack_bg - stack_ksz_tsz_fit
   # cutouts_dipole_ksz_tsz_corrected2 = []
   # for k in range(nber_clus):
   #     cutouts_dipole_ksz_tsz_corrected2.append(cutouts_aligned_ksz_tsz[k])
       
    covariance_matrix_baseline, correlation_matrix_baseline = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_baseline, 
                                                                                                weights_gradient_baseline, 
                                                                                                map_params, 
                                                                                                cutout_size_am = cutout_size_am, 
                                                                                                cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                l_cut = l_cut, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False, 
                                                                                                stack_bg = stack_bg)
    
    covariance_matrix_ksz, correlation_matrix_ksz = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_ksz, 
                                                                                                weights_gradient_ksz, 
                                                                                                map_params, 
                                                                                                cutout_size_am = cutout_size_am, 
                                                                                                cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                l_cut = l_cut, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False, 
                                                                                                stack_bg = stack_bg)
    
    covariance_matrix_tsz, correlation_matrix_tsz = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_tsz, 
                                                                                                weights_gradient_tsz, 
                                                                                                map_params, 
                                                                                                cutout_size_am = cutout_size_am, 
                                                                                                cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                l_cut = l_cut, l = l, cl = cl, 
                                                                                                cl_noise = cl_noise, 
                                                                                                correct_for_tsz = False, 
                                                                                                stack_bg = stack_bg)
    
   # covariance_matrix_tsz_corrected1, correlation_matrix_tsz_corrected1 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_tsz_corrected1, 
   #                                                                                             weights_gradient_tsz, 
   #                                                                                             map_params, 
   #                                                                                             cutout_size_am = cutout_size_am, 
   #                                                                                             cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
   #                                                                                             l_cut = l_cut, l = l, cl = cl, 
   #                                                                                             cl_noise = cl_noise, 
   #                                                                                             correct_for_tsz = False, 
   #                                                                                             stack_bg = stack_bg,
   #                                                                                             stack_tsz = stack_tsz)
    
   # covariance_matrix_tsz_corrected2, correlation_matrix_tsz_corrected2 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_tsz_corrected2, 
   #                                                                                             weights_gradient_tsz, 
   #                                                                                             map_params, 
   #                                                                                             cutout_size_am = cutout_size_am, 
   #                                                                                             cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
   #                                                                                             l_cut = l_cut, l = l, cl = cl, 
   #                                                                                             cl_noise = cl_noise, 
   #                                                                                             correct_for_tsz = False, 
   #                                                                                             stack_bg = stack_bg,
   #                                                                                             stack_tsz = stack_tsz_fit)
    
  #  covariance_matrix_ksz_tsz, correlation_matrix_ksz_tsz = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_ksz_tsz, 
  #                                                                                                              weights_gradient_ksz_tsz, 
  #                                                                                                              map_params, 
  #                                                                                                              cutout_size_am = cutout_size_am, 
  #                                                                                                              cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
  #                                                                                                              l_cut = l_cut, l = l, cl = cl, 
  #                                                                                                              cl_noise = cl_noise, 
  #                                                                                                              correct_for_tsz = False, 
  #                                                                                                              stack_bg = stack_bg)
    
  #  covariance_matrix_ksz_tsz_corrected1, correlation_matrix_ksz_tsz_corrected1 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_ksz_tsz_corrected1, 
  #                                                                                                                                    weights_gradient_ksz_tsz, 
  #                                                                                                                                    map_params, 
  #                                                                                                                                    cutout_size_am = cutout_size_am, 
  #                                                                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
  #                                                                                                                                    l_cut = l_cut, l = l, cl = cl, 
  #                                                                                                                                    cl_noise = cl_noise, 
  #                                                                                                                                    correct_for_tsz = False, 
  #                                                                                                                                    stack_bg = stack_bg,
   #                                                                                                                                   stack_tsz = stack_ksz_tsz)
    
  #  covariance_matrix_ksz_tsz_corrected2, correlation_matrix_ksz_tsz_corrected2 = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_ksz_tsz_corrected2, 
  #                                                                                                                                    weights_gradient_ksz_tsz, 
  #                                                                                                                                    map_params, 
  #                                                                                                                                    cutout_size_am = cutout_size_am, 
  #                                                                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
  #                                                                                                                                    l_cut = l_cut, l = l, cl = cl, 
  #                                                                                                                                    cl_noise = cl_noise, 
  #                                                                                                                                    correct_for_tsz = False, 
  #                                                                                                                                    stack_bg = stack_bg,
  #                                                                                                                                    stack_tsz = stack_ksz_tsz_fit)
    
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
    
    x_ksz, L_ksz, median_value_ksz, error_ksz = stats.run_ml(stack_dipole_ksz, models, covariance_matrix_ksz, mass_int, finer_reso = False)
    x_arr_ksz.append(x_ksz)
    L_arr_ksz.append(L_ksz)
    print(median_value_ksz, error_ksz) 
    M_ip_ksz, L_ip_ksz, median_value_finer_ksz, error_finer_ksz= stats.run_ml(stack_dipole_ksz, models, covariance_matrix_ksz, mass_int, finer_reso = True)
    M_ip_arr_ksz.append(M_ip_ksz)
    L_ip_arr_ksz.append(L_ip_ksz)
    print(median_value_finer_ksz, error_finer_ksz) 
    mass_arr_ksz.append(median_value_finer_ksz)
    
    x_tsz, L_tsz, median_value_tsz, error_tsz = stats.run_ml(stack_dipole_tsz, models, covariance_matrix_tsz, mass_int, finer_reso = False)
    x_arr_tsz.append(x_tsz)
    L_arr_tsz.append(L_tsz)
    print(median_value_tsz, error_tsz) 
    M_ip_tsz, L_ip_tsz, median_value_finer_tsz, error_finer_tsz= stats.run_ml(stack_dipole_tsz, models, covariance_matrix_tsz, mass_int, finer_reso = True)
    M_ip_arr_tsz.append(M_ip_tsz)
    L_ip_arr_tsz.append(L_ip_tsz)
    print(median_value_finer_tsz, error_finer_tsz) 
    mass_arr_tsz.append(median_value_finer_tsz)
    
 #   x_tsz_corrected1, L_tsz_corrected1, median_value_tsz_corrected1, error_tsz_corrected1 = stats.run_ml(stack_dipole_tsz_corrected1, models, covariance_matrix_tsz_corrected1, mass_int, finer_reso = False)
 #   x_arr_tsz_corrected1.append(x_tsz_corrected1)
 #   L_arr_tsz_corrected1.append(L_tsz_corrected1)
 #   print(median_value_tsz_corrected1, error_tsz_corrected1) 
 #   M_ip_tsz_corrected1, L_ip_tsz_corrected1, median_value_finer_tsz_corrected1, error_finer_tsz_corrected1 = stats.run_ml(stack_dipole_tsz_corrected1, models, covariance_matrix_tsz_corrected1, mass_int, finer_reso = True)
 #   M_ip_arr_tsz_corrected1.append(M_ip_tsz_corrected1)
 #   L_ip_arr_tsz_corrected1.append(L_ip_tsz_corrected1)
 #   print(median_value_finer_tsz_corrected1, error_finer_tsz_corrected1) 
 #   mass_arr_tsz_corrected1.append(median_value_finer_tsz_corrected1)
    
  #  x_arr_tsz_corrected2.append(x_tsz_corrected2)
  #  x_tsz_corrected2, L_tsz_corrected2, median_value_tsz_corrected2, error_tsz_corrected2 = stats.run_ml(stack_dipole_tsz_corrected2, models, covariance_matrix_tsz_corrected2, mass_int, finer_reso = False)
  #  print(median_value_tsz_corrected2, error_tsz_corrected2) 
  #  L_arr_tsz_corrected2.append(L_tsz_corrected2)
  #  M_ip_tsz_corrected2, L_ip_tsz_corrected2, median_value_finer_tsz_corrected2, error_finer_tsz_corrected2 = stats.run_ml(stack_dipole_tsz_corrected2, models, covariance_matrix_tsz_corrected2, mass_int, finer_reso = True)
  #  L_ip_arr_tsz_corrected2.append(L_ip_tsz_corrected2)
  #  M_ip_arr_tsz_corrected2.append(M_ip_tsz_corrected2)
  #  mass_arr_tsz_corrected2.append(median_value_finer_tsz_corrected2)
  #  print(median_value_finer_tsz_corrected2, error_finer_tsz_corrected2) 
    
  #  x_ksz_tsz, L_ksz_tsz, median_value_ksz_tsz, error_ksz_tsz = stats.run_ml(stack_dipole_ksz_tsz, models, covariance_matrix_ksz_tsz, mass_int, finer_reso = False)
  #  x_arr_ksz_tsz.append(x_ksz_tsz)
  #  L_arr_ksz_tsz.append(L_ksz_tsz)
  #  print(median_value_ksz_tsz, error_ksz_tsz) 
  #  M_ip_ksz_tsz, L_ip_ksz_tsz, median_value_finer_ksz_tsz, error_finer_ksz_tsz= stats.run_ml(stack_dipole_ksz_tsz, models, covariance_matrix_ksz_tsz, mass_int, finer_reso = True)
  #  M_ip_arr_ksz_tsz.append(M_ip_ksz_tsz)
  #  L_ip_arr_ksz_tsz.append(L_ip_ksz_tsz)
  #  print(median_value_finer_ksz_tsz, error_finer_ksz_tsz) 
  #  mass_arr_ksz_tsz.append(median_value_finer_ksz_tsz)
    
  #  x_ksz_tsz_corrected1, L_ksz_tsz_corrected1, median_value_ksz_tsz_corrected1, error_ksz_tsz_corrected1 = stats.run_ml(stack_dipole_ksz_tsz_corrected1, models, covariance_matrix_ksz_tsz_corrected1, mass_int, finer_reso = False)
  #  x_arr_ksz_tsz_corrected1.append(x_ksz_tsz_corrected1)
  #  L_arr_ksz_tsz_corrected1.append(L_ksz_tsz_corrected1)
  #  print(median_value_ksz_tsz_corrected1, error_ksz_tsz_corrected1) 
  #  M_ip_ksz_tsz_corrected1, L_ip_ksz_tsz_corrected1, median_value_finer_ksz_tsz_corrected1, error_finer_ksz_tsz_corrected1 = stats.run_ml(stack_dipole_ksz_tsz_corrected1, models, covariance_matrix_ksz_tsz_corrected1, mass_int, finer_reso = True)
  #  M_ip_arr_ksz_tsz_corrected1.append(M_ip_ksz_tsz_corrected1)
  #  L_ip_arr_ksz_tsz_corrected1.append(L_ip_ksz_tsz_corrected1)
  #  print(median_value_finer_ksz_tsz_corrected1, error_finer_ksz_tsz_corrected1) 
  #  mass_arr_ksz_tsz_corrected1.append(median_value_finer_ksz_tsz_corrected1)
    
  #  x_ksz_tsz_corrected2, L_ksz_tsz_corrected2, median_value_ksz_tsz_corrected2, error_ksz_tsz_corrected2 = stats.run_ml(stack_dipole_ksz_tsz_corrected2, models, covariance_matrix_ksz_tsz_corrected2, mass_int, finer_reso = False)
  #  x_arr_ksz_tsz_corrected2.append(x_ksz_tsz_corrected2)
  #  L_arr_ksz_tsz_corrected2.append(L_ksz_tsz_corrected2)
  #  print(median_value_ksz_tsz_corrected2, error_ksz_tsz_corrected2) 
  #  M_ip_ksz_tsz_corrected2, L_ip_ksz_tsz_corrected2, median_value_finer_ksz_tsz_corrected2, error_finer_ksz_tsz_corrected2 = stats.run_ml(stack_dipole_ksz_tsz_corrected2, models, covariance_matrix_ksz_tsz_corrected2, mass_int, finer_reso = True)
  #  M_ip_arr_ksz_tsz_corrected2.append(M_ip_ksz_tsz_corrected2)
  #  L_ip_arr_ksz_tsz_corrected2.append(L_ip_ksz_tsz_corrected2)
  #  print(median_value_finer_ksz_tsz_corrected2, error_finer_ksz_tsz_corrected2) 
  #  mass_arr_ksz_tsz_corrected2.append(median_value_finer_ksz_tsz_corrected2)

#%%
#likelihood_arr = [L_ip_arr_baseline, L_ip_arr_ksz, L_ip_arr_tsz, L_ip_arr_tsz_corrected1, L_ip_arr_tsz_corrected2, L_ip_arr_ksz_tsz, L_ip_arr_ksz_tsz_corrected1, L_ip_arr_ksz_tsz_corrected2]
#mass_results_arr = [mass_arr_baseline, mass_arr_ksz, mass_arr_tsz, mass_arr_tsz_corrected1, mass_arr_tsz_corrected2, mass_arr_ksz_tsz, mass_arr_ksz_tsz_corrected1, mass_arr_ksz_tsz_corrected2]

             
print('\n')
print('Combined Results:')    
x_comb_baseline, L_comb_baseline, median_value_comb_baseline, error_comb_baseline = stats.combined_likelihood(x_baseline, L_arr_baseline, normalize = True, finer_reso = False)
print(median_value_comb_baseline, error_comb_baseline) 
M_ip_comb_baseline, L_ip_comb_baseline, median_value_comb_finer_baseline, error_comb_finer_baseline = stats.combined_likelihood(x_baseline, L_arr_baseline, normalize = True, finer_reso = True)
print(median_value_comb_finer_baseline, error_comb_finer_baseline) 

x_comb_ksz, L_comb_ksz, median_value_comb_ksz, error_comb_ksz = stats.combined_likelihood(x_ksz, L_arr_ksz, normalize = True, finer_reso = False)
M_ip_comb_ksz, L_ip_comb_ksz, median_value_comb_finer_ksz, error_comb_finer_ksz = stats.combined_likelihood(x_ksz, L_arr_ksz, normalize = True, finer_reso = True)
print(median_value_comb_ksz, error_comb_ksz) 
print(median_value_comb_finer_ksz, error_comb_finer_ksz) 

x_comb_tsz, L_comb_tsz, median_value_comb_tsz, error_comb_tsz = stats.combined_likelihood(x_tsz, L_arr_tsz, normalize = True, finer_reso = False)
print(median_value_comb_tsz, error_comb_tsz) 
M_ip_comb_tsz, L_ip_comb_tsz, median_value_comb_finer_tsz, error_comb_finer_tsz = stats.combined_likelihood(x_tsz, L_arr_tsz, normalize = True, finer_reso = True)
print(median_value_comb_finer_tsz, error_comb_finer_tsz) 

#x_comb_tsz_corrected1, L_comb_tsz_corrected1, median_value_comb_tsz_corrected1, error_comb_tsz_corrected1 = stats.combined_likelihood(x_tsz_corrected1, L_arr_tsz_corrected1, normalize = True, finer_reso = False)
#print(median_value_comb_tsz_corrected1, error_comb_tsz_corrected1) 
#M_ip_comb_tsz_corrected1, L_ip_comb_tsz_corrected1, median_value_comb_finer_tsz_corrected1, error_comb_finer_tsz_corrected1 = stats.combined_likelihood(x_tsz_corrected1, L_arr_tsz_corrected1, normalize = True, finer_reso = True)
#print(median_value_comb_finer_tsz_corrected1, error_comb_finer_tsz_corrected1) 

#x_comb_tsz_corrected2, L_comb_tsz_corrected2, median_value_comb_tsz_corrected2, error_comb_tsz_corrected2 = stats.combined_likelihood(x_tsz_corrected2, L_arr_tsz_corrected2, normalize = True, finer_reso = False)
#print(median_value_comb_tsz_corrected2, error_comb_tsz_corrected2) 
#M_ip_comb_tsz_corrected2, L_ip_comb_tsz_corrected2, median_value_comb_finer_tsz_corrected2, error_comb_finer_tsz_corrected2 = stats.combined_likelihood(x_tsz_corrected2, L_arr_tsz_corrected2, normalize = True, finer_reso = True)
#print(median_value_comb_finer_tsz_corrected2, error_comb_finer_tsz_corrected2) 

#x_comb_ksz_tsz, L_comb_ksz_tsz, median_value_comb_ksz_tsz, error_comb_ksz_tsz = stats.combined_likelihood(x_ksz_tsz, L_arr_ksz_tsz, normalize = True, finer_reso = False)
#print(median_value_comb_ksz_tsz, error_comb_ksz_tsz) 
#M_ip_comb_ksz_tsz, L_ip_comb_ksz_tsz, median_value_comb_finer_ksz_tsz, error_comb_finer_ksz_tsz = stats.combined_likelihood(x_ksz_tsz, L_arr_ksz_tsz, normalize = True, finer_reso = True)
#print(median_value_comb_finer_ksz_tsz, error_comb_finer_ksz_tsz) 

#x_comb_ksz_tsz_corrected1, L_comb_ksz_tsz_corrected1, median_value_comb_ksz_tsz_corrected1, error_comb_ksz_tsz_corrected1 = stats.combined_likelihood(x_ksz_tsz_corrected1, L_arr_ksz_tsz_corrected1, normalize = True, finer_reso = False)
#print(median_value_comb_ksz_tsz_corrected1, error_comb_ksz_tsz_corrected1) 
#M_ip_comb_ksz_tsz_corrected1, L_ip_comb_ksz_tsz_corrected1, median_value_comb_finer_ksz_tsz_corrected1, error_comb_finer_ksz_tsz_corrected1 = stats.combined_likelihood(x_ksz_tsz_corrected1, L_arr_ksz_tsz_corrected1, normalize = True, finer_reso = True)
#print(median_value_comb_finer_ksz_tsz_corrected1, error_comb_finer_ksz_tsz_corrected1) 

#x_comb_ksz_tsz_corrected2, L_comb_ksz_tsz_corrected2, median_value_comb_ksz_tsz_corrected2, error_comb_ksz_tsz_corrected2 = stats.combined_likelihood(x_ksz_tsz_corrected2, L_arr_ksz_tsz_corrected2, normalize = True, finer_reso = False)
#M_ip_comb_ksz_tsz_corrected2, L_ip_comb_ksz_tsz_corrected2, median_value_comb_finer_ksz_tsz_corrected2, error_comb_finer_ksz_tsz_corrected2 = stats.combined_likelihood(x_ksz_tsz_corrected2, L_arr_ksz_tsz_corrected2, normalize = True, finer_reso = True)
#print(median_value_comb_ksz_tsz_corrected2, error_comb_ksz_tsz_corrected2) 
#print(median_value_comb_finer_ksz_tsz_corrected2, error_comb_finer_ksz_tsz_corrected2) 


#comb_likelihood_arr = [L_ip_comb_baseline, L_ip_comb_ksz, L_ip_comb_tsz, L_ip_comb_tsz_corrected1, L_ip_comb_tsz_corrected2,
#                       L_ip_comb_ksz_tsz, L_ip_comb_ksz_tsz_corrected1, L_ip_comb_ksz_tsz_corrected2]
#comb_mass_arr = [median_value_comb_finer_baseline, median_value_comb_finer_ksz, 
#                 median_value_comb_finer_tsz, median_value_comb_finer_tsz_corrected1, median_value_comb_finer_tsz_corrected2,
#                 median_value_comb_finer_ksz_tsz, median_value_comb_finer_ksz_tsz_corrected1, median_value_comb_finer_ksz_tsz_corrected2]
#comb_error_arr = [error_comb_finer_baseline, error_comb_finer_ksz, 
#                  error_comb_finer_tsz, error_comb_finer_tsz_corrected1, error_comb_finer_tsz_corrected2,
#                  error_comb_finer_ksz_tsz, error_comb_finer_ksz_tsz_corrected1, error_comb_finer_ksz_tsz_corrected2]


#%%

#np.save('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_likelihoods.npy', likelihood_arr)
#np.save('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_likelihoods.npy', comb_likelihood_arr)
#np.save('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_median_masses.npy', mass_results_arr)
#np.save('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_median_masses.npy', comb_mass_arr)
#np.save('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_errors.npy', comb_error_arr)

#combined_mass_results_file = ''
#for i in range(len(comb_mass_arr)):
#    combined_mass_results_file += "{0:11.3f}{1:>1}{2:5.3f}".format(comb_mass_arr[i], "+-", comb_error_arr[i]) + "\n"  

#file = open('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_combined_mass_results.txt',"w")
#file.write(combined_mass_results_file)
#file.close()


###################################################################################################################################
#%%

"""
plt.imshow(stack_tsz)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_tsz_fit)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_dipole_tsz)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_dipole_tsz_corrected1)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_dipole_tsz_corrected2)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_ksz_tsz)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_ksz_tsz_fit)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(stack_dipole_ksz_tsz)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(covariance_matrix_ksz_tsz_corrected1)#stack_dipole_ksz_tsz_corrected1)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(covariance_matrix_ksz_tsz_corrected2)#stack_dipole_ksz_tsz_corrected2)
plt.colorbar()
plt.show()
"""

#%%
#likelihood_arr = np.load('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_likelihoods.npy')
#comb_likelihood_arr = np.load('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_likelihoods.npy')
nber_runs_exp = 5
start = 1000
end = 2000
for i in [3,4]:#range(nber_runs_exp):   
    plt.plot(x_arr_baseline[i][100:200], L_arr_baseline[i][100:200], ls = '-', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_baseline[i][start:end], L_ip_arr_baseline[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_baseline[100:200], L_comb_baseline[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_baseline[start:end], L_ip_comb_baseline[start:end], ls = '-', marker = '', color = 'red')  
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in [3,4]:   
    plt.plot(x_arr_ksz[i][100:200], L_arr_ksz[i][100:200], ls = '-', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_ksz[i][start:end], L_ip_arr_ksz[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_ksz[100:200], L_comb_ksz[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_ksz[start:end], L_ip_comb_ksz[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in [3,4]:  
    plt.plot(x_arr_tsz[i][100:200], L_arr_tsz[i][100:200], ls = '-', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_tsz[i][start:end], L_ip_arr_tsz[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_tsz[100:200], L_comb_tsz[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_tsz[start:end], L_ip_comb_tsz[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
"""
for i in range(nber_runs_exp):   
    plt.plot(x_arr_tsz_corrected1[i][100:200], L_arr_tsz_corrected1[i][100:200], ls = '', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_tsz_corrected1[i][start:end], L_ip_arr_tsz_corrected1[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_tsz_corrected1[100:200], L_comb_tsz_corrected1[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_tsz_corrected1[start:end], L_ip_comb_tsz_corrected1[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in range(nber_runs_exp):   
    plt.plot(x_arr_tsz_corrected2[i][100:200], L_arr_tsz_corrected2[i][100:200], ls = '', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_tsz_corrected2[i][start:end], L_ip_arr_tsz_corrected2[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_tsz_corrected2[100:200], L_comb_tsz_corrected2[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_tsz_corrected2[start:end], L_ip_comb_tsz_corrected2[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in range(nber_runs_exp):   
    plt.plot(x_arr_ksz_tsz[i][100:200], L_arr_ksz_tsz[i][100:200], ls = '', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_ksz_tsz[i][start:end], L_ip_arr_ksz_tsz[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_ksz_tsz[100:200], L_comb_ksz_tsz[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_ksz_tsz[start:end], L_ip_comb_ksz_tsz[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in range(nber_runs_exp):   
    plt.plot(x_arr_ksz_tsz_corrected1[i][100:200], L_arr_ksz_tsz_corrected1[i][100:200], ls = '', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_ksz_tsz_corrected1[i][start:end], L_ip_arr_ksz_tsz_corrected1[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_ksz_tsz_corrected1[100:200], L_comb_ksz_tsz_corrected1[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_ksz_tsz_corrected1[start:end], L_ip_comb_ksz_tsz_corrected1[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
plt.subplots()
for i in range(nber_runs_exp):   
    plt.plot(x_arr_ksz_tsz_corrected2[i][100:200], L_arr_ksz_tsz_corrected2[i][100:200], ls = '', marker = 'x', color = 'blue')
    plt.plot(M_ip_arr_ksz_tsz_corrected2[i][start:end], L_ip_arr_ksz_tsz_corrected2[i][start:end], ls = '-', marker = '', color = 'deepskyblue', alpha = 0.25)
plt.plot(x_comb_ksz_tsz_corrected2[100:200], L_comb_ksz_tsz_corrected2[100:200], ls = '', marker = 'o', color = 'orange')
plt.plot(M_ip_comb_ksz_tsz_corrected2[start:end], L_ip_comb_ksz_tsz_corrected2[start:end], ls = '-', marker = '', color = 'red')   
plt.axvline(1.47, color = 'green')
plt.show()
"""
#plt.savefig('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_check.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)    


#%%
"""
#mass_results_arr = np.load('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_median_masses.npy')
#comb_mass_arr = np.load('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_median_masses.npy')
#comb_error_arr = np.load('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias_comb_errors.npy')

mass_results_arr = [mass_results_arr[0], mass_results_arr[1], mass_results_arr[2], mass_results_arr[3], mass_results_arr[6]]
comb_mass_arr = [comb_mass_arr[0], comb_mass_arr[1], comb_mass_arr[2], comb_mass_arr[3], comb_mass_arr[6]]
comb_error_arr = [comb_error_arr[0], comb_error_arr[1], comb_error_arr[2], comb_error_arr[3], comb_error_arr[6]]

fig, ax = plt.subplots(figsize=(8,6))
xvals = [0.05, 0.05, 0.05, 0.05, 0.05]
yvals = [0.15, 0.35, 0.55, 0.75, 0.95]
titles_arr = ['baseline', '+ kSZ', '+ tSZ', '+ tSZ corrected', r'+ kSZ & tSZ corrected']
ax.set_ylim(0.,1)
markers = ['o', '^', 'd', 'v', 'D']
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
    ax.errorbar((np.round(comb_mass_arr[i], 3)-1.47), yvals[i], xerr = np.round(comb_error_arr[i],3), elinewidth = 4, color = color_arr[i],  marker = markers[i], markersize = 12, ls = '', capsize=5, capthick=3)

#plt.savefig('/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)

"""

