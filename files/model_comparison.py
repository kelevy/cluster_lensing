#%%
import numpy as np
import matplotlib.pylab as plt
import utils
import cosmo
import mockobs
import lensing_estimator


#%%
nber_clus = 2500
nber_rand = 50000
cutout_size_am = 10
cutout_size_for_grad_est_am = 6
l_cut = 2000
average = 1
nber_runs = 20
map_params = [120, 0.5, 120, 0.5] # nx, dx, ny, dy
lensing_reso = 5
beam_fwhm = 1.0 # arcmin
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
z = 0.7
mass_int = [2] 
nx = 120
ny = 120
dx = 0.5
dy = 0.5


#%%
print('getting models')
model1 = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
                                   cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                   l_cut = l_cut, lensing_reso = lensing_reso, average = average)[0]

  
model2 = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
                                   cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                   l_cut = l_cut, lensing_reso = lensing_reso, average = average)[0]


#%%
plt.imshow(model1)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(model2)
plt.colorbar()
plt.show()
plt.subplots()
plt.imshow(model2-model1)
plt.colorbar()
plt.show()


#%%
print('getting background')
maps_rand = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand, lensing_reso = lensing_reso)
stack_bg, cutouts_aligned_rand, weights_gradient_rand = lensing_estimator.get_aligned_stack(maps_rand, map_params, l = l, cl = cl, 
                                                                                            cl_noise = cl_noise, cutout_size_am = cutout_size_am, 
                                                                                            cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                            l_cut = l_cut)  
del maps_rand
    
    
#%%
 
print('getting maps')
maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [2e14, cosmo.concentration_parameter(2e14, z, 0.674), z], nber_obs = nber_clus, lensing_reso = lensing_reso)

for j in range(nber_clus):
    white_noise_map = utils.make_gaussian_realization(l, nl, map_params) 
    maps_clus[j] = utils.convolve(maps_clus[j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map

print('getting stack') 
stack_aligned_baseline, cutouts_aligned_baseline, weights_gradient_baseline = lensing_estimator.get_aligned_stack(maps_clus, 
                                                                                                                  map_params, 
                                                                                                                  l = l, cl = cl, 
                                                                                                                  cl_noise = cl_noise, 
                                                                                                                  cutout_size_am = cutout_size_am, 
                                                                                                                  cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                                  l_cut = l_cut)      


stack_dipole_baseline = stack_aligned_baseline - stack_bg
cutouts_dipole_baseline = []
for k in range(nber_clus):
    cutouts_dipole_baseline.append(cutouts_aligned_baseline[k])
      

print('getting covariance')    
covariance_matrix_baseline, correlation_matrix_baseline = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_baseline, 
                                                                                                              weights_gradient_baseline, 
                                                                                                              map_params, 
                                                                                                              cutout_size_am = cutout_size_am, 
                                                                                                              cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                              l_cut = l_cut, l = l, cl = cl, 
                                                                                                              cl_noise = cl_noise, 
                                                                                                              correct_for_tsz = False, 
                                                                                                              stack_bg = stack_bg)
 

#%% 
print('getting likelihood value')
import scipy as sp
cov=np.mat(covariance_matrix_baseline)
cov_inv=sp.linalg.pinv(cov)

d=stack_dipole_baseline.flatten()
m=model1.flatten()
d=d-m
#diff = (stack_dipole_baseline - model1).flatten()
#chi_2 = np.dot(diff.T, np.dot(np.linalg.pinv(cov), diff))
# logLval =  -0.5*chi_2
logLval= -0.5 * np.asarray( np.dot(d.T, np.dot( cov_inv, d ))).squeeze()
#L=np.exp(lnlike)
print(logLval)

d=stack_dipole_baseline.flatten()
m=model2.flatten()
d=d-m
#diff = (stack_dipole_baseline - model2).flatten()
#chi_2 = np.dot(diff.T, np.dot(np.linalg.pinv(cov), diff))
# logLval =  -0.5*chi_2
logLval= -0.5 * np.asarray( np.dot(d.T, np.dot( cov_inv, d ))).squeeze()
#L=np.exp(lnlike)
print(logLval)
