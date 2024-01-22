%load_ext autoreload 
%autoreload 2
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from pylab import rcParams
from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
rcParams['font.family'] = 'serif'
from mpl_toolkits.axes_grid1 import make_axes_locatable
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
from tqdm import tqdm
import cosmo
import lensing_estimator
import experiments as exp
import sims
import stats

nber_clus = 2500
nber_rand = 50000
nber_runs = 20
nx, ny = 120, 120
dx, dy = 0.5, 0.5
map_params = [nx, dx, ny, dy] 
beam_fwhm = 1.0 # arcmin
noiseval_white = 2.0 # uK-arcmin
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = exp.beam_power_spectrum(beam_fwhm, l)
nl = exp.instrumental_noise_power_spectrum(noiseval_white, l)
cl_noise = exp.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
z = 0.7
mass_int = np.arange(0, 4, 0.01)
M_input = 2.0e14

#%%script false --no-raise-error
for i in range(5):    
    models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, bl = bl, cl_noise = cl_noise, average = 1, cutout_size_am = 60)
    np.save('results/models_60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version'+str(i)+'.npy', models)
    
#%%script false --no-raise-error
for i in range(5):    
    models = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, bl = bl, cl_noise = cl_noise, average = 1, cutout_size_am = 10)
    np.save('results/models_10arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version'+str(i)+'.npy', models)
    
models_arr_60arcmin = []
for i in range(5):    
    models = np.load('results/models_60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version'+str(i)+'.npy')  
    models_arr_60arcmin.append(models)
plt.imshow(models[-1])
plt.colorbar()
plt.subplots()

models_arr_60arcmin = []
for i in range(5):    
    models = np.load('results/models_60arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version'+str(i)+'.npy') 
    models_small = []
    for model in models:
        models_small.append(model[50:70, 50:70])
    models = np.copy(models_small)
    models_arr_60arcmin.append(models)
plt.imshow(models[-1])
plt.colorbar()

models_arr_10arcmin = []
for i in range(5):    
    models = np.load('results/models_10arcmin_massint0to4_massres0.01_2.0uKarcmin_1.0arcmin_2500clus_version'+str(i)+'.npy')  
    models_arr_10arcmin.append(models)
plt.imshow(models[-1])
plt.colorbar()

plt.imshow(models_arr_60arcmin[0][-1])
plt.colorbar()
plt.show()
plt.imshow(models_arr_60arcmin[1][-1])
plt.colorbar()
plt.show()
plt.imshow(models_arr_60arcmin[0][-1]-models_arr_60arcmin[1][-1])
plt.colorbar()

plt.imshow(models_arr_10arcmin[0][-1])
plt.colorbar()
plt.show()
plt.imshow(models_arr_10arcmin[1][-1])
plt.colorbar()
plt.show()
plt.imshow(models_arr_10arcmin[0][-1]-models_arr_10arcmin[1][-1])
plt.colorbar()

#%%script false --no-raise-error
maps_rand = sims.cmb_mock_data(map_params, l, cl,  bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg, _, _ = lensing_estimator.background(maps_rand, map_params, l = l, cl = cl, cl_noise = cl_noise, cutout_size_am = 60) 
del maps_rand
np.save('results/background_60arcmin_2.0uKarcmin_1.0arcmin_50000rand.npy', stack_bg)


#%%script false --no-raise-error
maps_rand = sims.cmb_mock_data(map_params, l, cl,  bl = bl, nl = nl, nber_obs = nber_rand)
stack_bg, _, _ = lensing_estimator.background(maps_rand, map_params, l = l, cl = cl, cl_noise = cl_noise, cutout_size_am = 10) 
del maps_rand
np.save('results/background_10arcmin_2.0uKarcmin_1.0arcmin_50000rand.npy', stack_bg)

stack_bg_60arcmin = np.load('results/background_60arcmin_2.0uKarcmin_1.0arcmin_50000rand.npy')
plt.imshow(stack_bg_60arcmin)
plt.colorbar()

stack_bg_10arcmin = np.load('results/background_10arcmin_2.0uKarcmin_1.0arcmin_50000rand.npy')
plt.imshow(stack_bg_10arcmin)
plt.colorbar()

lnbaseline_arr_60arcmin = []
baseline_arr_60arcmin = []

lnbaseline_arr_10arcmin = []
baseline_arr_10arcmin = []

for i in range(nber_runs):    
    lnbaseline_arr_temp_60arcmin = []
    baseline_arr_temp_60arcmin = []
    lnbaseline_arr_temp_10arcmin = []
    baseline_arr_temp_10arcmin = []
    
    maps_clus_baseline = sims.cmb_mock_data(map_params, l, cl, cluster = [M_input, cosmo.concentration_parameter(M_input, z, 0.674), z], bl = bl, nl = nl, nber_obs = nber_clus)
    
    stack_dipole_baseline_60arcmin, cutouts_dipole_baseline_60arcmin, magnitudes_dipole_baseline_60arcmin = lensing_estimator.dipole(maps_clus_baseline, stack_bg_60arcmin, map_params, l = l, cl = cl, cl_noise = cl_noise, cutout_size_am = 60)     
    stack_dipole_baseline_10arcmin, cutouts_dipole_baseline_10arcmin, magnitudes_dipole_baseline_10arcmin = lensing_estimator.dipole(maps_clus_baseline, stack_bg_10arcmin, map_params, l = l, cl = cl, cl_noise = cl_noise, cutout_size_am = 10) 
    
    stack_dipole_baseline_small = stack_dipole_baseline_60arcmin[50:70, 50:70]
    stack_dipole_baseline_60arcmin = np.copy(stack_dipole_baseline_small)
    cutouts_dipole_baseline_small = []
    for dipole in cutouts_dipole_baseline_60arcmin:
        cutouts_dipole_baseline_small.append(dipole[50:70, 50:70])
    cutouts_dipole_baseline_60arcmin = np.copy(cutouts_dipole_baseline_small)   
    
    covariance_matrix_baseline_60arcmin, correlation_matrix_baseline_60arcmin = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_baseline_60arcmin, magnitudes_dipole_baseline_60arcmin)
    covariance_matrix_baseline_10arcmin, correlation_matrix_baseline_10arcmin = lensing_estimator.covariance_and_correlation_matrix(cutouts_dipole_baseline_10arcmin, magnitudes_dipole_baseline_10arcmin)
    
    data_baseline_60arcmin = stack_dipole_baseline_60arcmin, covariance_matrix_baseline_60arcmin
    data_baseline_10arcmin = stack_dipole_baseline_10arcmin, covariance_matrix_baseline_10arcmin
    
    print('Run', i+1)
    for k in range(len(models_arr_60arcmin)):
        lnlike_baseline_60arcmin, M_ip_baseline_60arcmin, L_ip_baseline_60arcmin, median_value_baseline_60arcmin, error_baseline_60arcmin = stats.run_ml(data_baseline_60arcmin, models_arr_60arcmin[k], mass_int)
        lnlike_baseline_10arcmin, M_ip_baseline_10arcmin, L_ip_baseline_10arcmin, median_value_baseline_10arcmin, error_baseline_10arcmin = stats.run_ml(data_baseline_10arcmin, models_arr_10arcmin[k], mass_int)
        lnbaseline_arr_temp_60arcmin.append(lnlike_baseline_60arcmin)
        lnbaseline_arr_temp_10arcmin.append(lnlike_baseline_10arcmin)
        baseline_arr_temp_60arcmin.append(L_ip_baseline_60arcmin)
        baseline_arr_temp_10arcmin.append(L_ip_baseline_10arcmin)
        print(median_value_baseline_60arcmin, error_baseline_60arcmin)
        print(median_value_baseline_10arcmin, error_baseline_10arcmin)
    lnbaseline_arr_60arcmin.append(lnbaseline_arr_temp_60arcmin)
    lnbaseline_arr_10arcmin.append(lnbaseline_arr_temp_10arcmin)
    baseline_arr_60arcmin.append(baseline_arr_temp_60arcmin)
    baseline_arr_10arcmin.append(baseline_arr_temp_10arcmin)

sorted_lnbaseline_arr_60arcmin = []
sorted_lnbaseline_arr_10arcmin = []
for i in range(len(models_arr_60arcmin)):
    sorted_lnbaseline_arr_temp_60arcmin = []
    sorted_lnbaseline_arr_temp_10arcmin = []
    for j in range(nber_runs):
        ln_60arcmin = lnbaseline_arr_60arcmin[j][i] 
        ln_10arcmin = lnbaseline_arr_10arcmin[j][i] 
        sorted_lnbaseline_arr_temp_60arcmin.append(ln_60arcmin)
        sorted_lnbaseline_arr_temp_10arcmin.append(ln_10arcmin)
    sorted_lnbaseline_arr_60arcmin.append(sorted_lnbaseline_arr_temp_60arcmin)
    sorted_lnbaseline_arr_10arcmin.append(sorted_lnbaseline_arr_temp_10arcmin)
    
sorted_baseline_arr_60arcmin = []
sorted_baseline_arr_10arcmin = []
for i in range(len(models_arr_60arcmin)):
    sorted_baseline_arr_temp_60arcmin = []
    sorted_baseline_arr_temp_10arcmin = []
    for j in range(nber_runs):
        lk_60arcmin = baseline_arr_60arcmin[j][i] 
        lk_10arcmin = baseline_arr_10arcmin[j][i]
        sorted_baseline_arr_temp_60arcmin.append(lk_60arcmin)
        sorted_baseline_arr_temp_10arcmin.append(lk_10arcmin)
    sorted_baseline_arr_60arcmin.append(sorted_baseline_arr_temp_60arcmin)
    sorted_baseline_arr_10arcmin.append(sorted_baseline_arr_temp_10arcmin)
        
print('\n')   
print('Combined Results:')
comb_lk_arr_60arcmin = []
comb_lk_arr_10arcmin = []
combined_mass_results_file_60arcmin = ''
combined_mass_results_file_10arcmin = ''
for i in range(len(models_arr_60arcmin)):
    _, x, comb_likelihood_baseline_60arcmin, comb_mass_est_baseline_60arcmin, comb_error_baseline_60arcmin = stats.combined_likelihood(mass_int, sorted_lnbaseline_arr_60arcmin[i])
    _, x, comb_likelihood_baseline_10arcmin, comb_mass_est_baseline_10arcmin, comb_error_baseline_10arcmin = stats.combined_likelihood(mass_int, sorted_lnbaseline_arr_10arcmin[i])
    comb_lk_arr_60arcmin.append(comb_likelihood_baseline_60arcmin)
    comb_lk_arr_10arcmin.append(comb_likelihood_baseline_10arcmin)
    combined_mass_results_file_60arcmin += "{0:11.3f}{1:>1}{2:5.3f}".format(comb_mass_est_baseline_60arcmin, "+-", comb_error_baseline_60arcmin) + "\n"
    combined_mass_results_file_10arcmin += "{0:11.3f}{1:>1}{2:5.3f}".format(comb_mass_est_baseline_10arcmin, "+-", comb_error_baseline_10arcmin) + "\n" 
    print(comb_mass_est_baseline_60arcmin, comb_error_baseline_60arcmin)
    print(comb_mass_est_baseline_10arcmin, comb_error_baseline_10arcmin)
    
file_60arcmin = open('results/combined_mass_results_model_bias_60arcmin.txt',"w")
file_60arcmin.write(combined_mass_results_file_60arcmin)
file_60arcmin.close()

file_10arcmin = open('results/combined_mass_results_model_bias_10arcmin.txt',"w")
file_10arcmin.write(combined_mass_results_file_10arcmin)
file_10arcmin.close()


fig, axs = plt.subplots(5, 1, figsize = (15, 30))
titles_arr = ['60armcin Model '+str(i) for i in range(len(models_arr_60arcmin))]

for i in range(len(axs)):
    for j in range(nber_runs):
        axs[i].set_title(titles_arr[i], size = 20)
        axs[i].plot(M_ip_baseline_60arcmin, sorted_baseline_arr_60arcmin[i][j], color = 'red', alpha = 0.2)
    axs[i].axvline(2.0, color = 'green')
    axs[i].plot(x, comb_lk_arr_60arcmin[i], color = 'black')

plt.savefig('results/result_model_bias_60arcmin.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)


fig, axs = plt.subplots(5, 1, figsize = (15, 30))
titles_arr = ['10arcmin Model '+str(i) for i in range(len(models_arr_10arcmin))]

for i in range(len(axs)):
    for j in range(nber_runs):
        axs[i].set_title(titles_arr[i], size = 20)
        axs[i].plot(M_ip_baseline_10arcmin, sorted_baseline_arr_10arcmin[i][j], color = 'red', alpha = 0.2)
    axs[i].axvline(2.0, color = 'green')
    axs[i].plot(x, comb_lk_arr_10arcmin[i], color = 'black')

plt.savefig('results/result_model_bias_10arcmin.pdf', dpi = 200., bbox_inches = 'tight', pad_inches = 0.1)