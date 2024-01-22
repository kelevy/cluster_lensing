# importing relevant modules
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
#from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rcParams['font.family'] = 'serif'
#rc('text.latex',preamble=r'\\usepackage{/Users/kevinlevy/Documents/cmb_cluster_lensing/apjfonts}')
from mpl_toolkits.axes_grid1 import make_axes_locatable
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import lensing_estimator
import mockobs
import stats
import utils


####################################################################################################################################
#%%


# defining relevant parameters
nber_clus = 2500
nber_rand = 50000
cutout_size_am = 10
cutout_size_for_grad_est_am = 6
l_cut = 2000
average = 20
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
mass_reso = 0.01
mass_min = 1.0
mass_max = 2.0
mass_int = np.arange(mass_min, mass_max, mass_reso)
M_input = 1.47e14
save_loc = '/Volumes/Extreme_SSD/cluster_lensing/results/foreground_bias'
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 18
labelsize = 15


###################################################################################################################################
#%%


# getting kSZ maps
print('Getting kSZ maps ...')
fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/ksz_tsz/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
ksz_cutouts = np.load(fname)

nx_cutout, ny_cutout = ksz_cutouts[0].shape[0], ksz_cutouts[0].shape[1]
s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
 
#%%


# checking ksz maps
ksz_check = np.mean(ksz_cutouts[:2500], axis = 0)[50:70, 50:70]
plt.imshow(ksz_check)
plt.colorbar()
plt.show()
plt.subplots()


###################################################################################################################################
#%%
cmb_maps = []
noise_maps = []
lensing_reso = 1
map_params_lensing = [map_params[0]*lensing_reso, map_params[1]/lensing_reso, map_params[2]*lensing_reso, map_params[3]/lensing_reso]
print(map_params_lensing)
print(int(nber_clus*average))
for i in range(int(nber_clus*average)):
    sim = mockobs.cmb_mock_data(map_params_lensing, l, cl, lensing_reso = lensing_reso)
    noise_map = utils.make_gaussian_realization(l, nl, map_params)
    cmb_maps.append(sim)
    noise_maps.append(noise_map)

# generating models
print('Generating models ...')
#models_baseline = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
 #                                 cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
#                                  l_cut = l_cut, average = average, cmb_maps = cmb_maps, noise_maps = noise_maps, ksz_cutouts = None)
#np.save('models_baseline2', models_baseline)
models_baseline = np.load('models_baseline.npy')
#models_ksz = lensing_estimator.models(nber_clus, map_params, l, cl, mass_int, z, centroid_shift_value = 0, bl = bl, nl = nl,
#                                  cl_noise = cl_noise, cutout_size_am = cutout_size_am, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
#                                  l_cut = l_cut, average = average, cmb_maps = cmb_maps, noise_maps = noise_maps, ksz_cutouts = ksz_cutouts)
models_ksz = np.load('models_ksz2.npy')
#models = np.load(save_loc+'/models_10arcmin_massint'+str(mass_min)+'to'+str(mass_max)+'_massres'+str(mass_reso)+'_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_clus)+'clus_avg'+str(average)+'.npy')


#%%

# checking models
models_check = [models_baseline[0], models_baseline[int(len(mass_int)/2)], models_baseline[-1]]
_, extent = utils.make_grid([models_check[0].shape[0], reso_arcmin, models_check[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(models_check), 1, figsize=(10,10))
dec = len(str(mass_reso).split(".")[1])
titles = ['Model '+str(np.round(mass_int[0], dec)), 'Model '+str(np.round(mass_int[int(len(mass_int)/2)], dec)), 'Model '+str(np.round(mass_int[-1], dec))]

for i in range(len(axarr)):
    titstr = titles[i]
    im = axarr[i].imshow(models_check[i], extent = extent, cmap = cm.RdBu_r)
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

models_check = [models_ksz[0], models_ksz[int(len(mass_int)/2)], models_ksz[-1]]
_, extent = utils.make_grid([models_check[0].shape[0], reso_arcmin, models_check[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(models_check), 1, figsize=(10,10))
dec = len(str(mass_reso).split(".")[1])
titles = ['Model '+str(np.round(mass_int[0], dec)), 'Model '+str(np.round(mass_int[int(len(mass_int)/2)], dec)), 'Model '+str(np.round(mass_int[-1], dec))]

for i in range(len(axarr)):
    titstr = titles[i]
    im = axarr[i].imshow(models_check[i], extent = extent, cmap = cm.RdBu_r)
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

models_check = [models_ksz[0]-models_baseline[0], models_ksz[int(len(mass_int)/2)]-models_baseline[int(len(mass_int)/2)], models_ksz[-1]-models_baseline[-1]]
_, extent = utils.make_grid([models_check[0].shape[0], reso_arcmin, models_check[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(models_check), 1, figsize=(10,10))
dec = len(str(mass_reso).split(".")[1])
titles = ['Model '+str(np.round(mass_int[0], dec)), 'Model '+str(np.round(mass_int[int(len(mass_int)/2)], dec)), 'Model '+str(np.round(mass_int[-1], dec))]

for i in range(len(axarr)):
    titstr = titles[i]
    im = axarr[i].imshow(models_check[i], extent = extent, cmap = cm.RdBu_r)
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

models = [models_baseline, models_baseline, models_ksz]

###################################################################################################################################
#%%


# generating background stack
print('Generating background stack ...')
stack_bg = np.load(save_loc+'/background_stack_10arcmin_'+str(beam_fwhm)+'arcmin_'+str(noiseval_white)+'uKarcmin_'+str(nber_rand)+'sims.npy') 

#%%

# checking background_stack
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


###################################################################################################################################
#%%

# starting analysis
x_arr = []
L_arr = []
M_ip_arr = []
L_ip_arr = []
median_mass_arr = []


for i in tqdm(range(nber_runs)):    
    x_arr_curr = []
    L_arr_curr = []
    M_ip_arr_curr = []
    L_ip_arr_curr = []
    median_mass_arr_curr = []    
    
    # generating cluster maps
    maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [M_input, cosmo.concentration_parameter(M_input, z, 0.674), z], nber_obs = nber_clus)
    maps = [np.copy(maps_clus), np.copy(maps_clus)]  
    for j in range(nber_clus):
        rand_sel = random.randint(0, len(ksz_cutouts)-1)
        white_noise_map = utils.make_gaussian_realization(l, nl, map_params) 
        maps[0][j] = utils.convolve(maps[0][j], l, np.sqrt(bl), mapparams = map_params) + white_noise_map
        maps[1][j] = utils.convolve(maps[1][j] + ksz_cutouts[rand_sel], l, np.sqrt(bl), mapparams = map_params) + white_noise_map

    # computing cluster stacks   
    stacks_aligned, cutouts_aligned_arr, weights_gradient_arr = [], [], []
    for  j in range(len(maps)):
        stack_aligned, cutouts_aligned, weights_gradient = lensing_estimator.get_aligned_stack(maps[j], 
                                                                                               map_params, 
                                                                                               l = l, cl = cl, 
                                                                                               cl_noise = cl_noise, 
                                                                                               cutout_size_am = cutout_size_am, 
                                                                                               cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                               l_cut = l_cut)  
        stacks_aligned.append(stack_aligned)
        cutouts_aligned_arr.append(cutouts_aligned)
        weights_gradient_arr.append(weights_gradient)                                                         

    # computing dipole stacks
    stacks_dipole = []
    for j in range(len(maps)):
        stack_dipole = stacks_aligned[j] - stack_bg 
        stacks_dipole.append(stack_dipole)            
    stacks_dipole.append(stack_dipole)
    
    # computing covariance matrices
    covariance_matrix_arr, correlation_matrix_arr = [], []
    for j in range(len(maps)):  
        covariance_matrix, correlation_matrix = lensing_estimator.covariance_and_correlation_matrix(cutouts_aligned_arr[j], 
                                                                                                    weights_gradient_arr[j], 
                                                                                                    map_params, 
                                                                                                    cutout_size_am = cutout_size_am, 
                                                                                                    cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, 
                                                                                                    l_cut = l_cut, l = l, cl = cl, 
                                                                                                    cl_noise = cl_noise, 
                                                                                                    stack_bg = stack_bg,
                                                                                                    stack_tsz = None)
        covariance_matrix_arr.append(covariance_matrix)
        correlation_matrix_arr.append(correlation_matrix)
    covariance_matrix_arr.append(covariance_matrix)
    correlation_matrix_arr.append(correlation_matrix)
        
    # getting likelihoods and mass estimates
    print('Run', i+1)
    for j in range(len(stacks_dipole)):  
        x, L, median_value, error = stats.run_ml(stacks_dipole[j], models_ksz, covariance_matrix_arr[j], mass_int, finer_reso = False)
        x_arr_curr.append(x)
        L_arr_curr.append(L)
        print(median_value, error) 
        M_ip, L_ip, median_value_finer, error_finer = stats.run_ml(stacks_dipole[j], models_ksz, covariance_matrix_arr[j], mass_int, finer_reso = True)
        M_ip_arr_curr.append(M_ip)
        L_ip_arr_curr.append(L_ip)
        median_mass_arr_curr.append(median_value_finer)
        print(median_value_finer, error_finer) 
       
    x_arr.append(x_arr_curr)
    L_arr.append(L_arr_curr)
    M_ip_arr.append(M_ip_arr_curr)
    L_ip_arr.append(L_ip_arr_curr)
    median_mass_arr.append(median_mass_arr_curr)
    
#%%

# sorting individual runs after case
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
    
# getting combined likelihoods and mass estimates          
print('\n')
print('Combined Results:')  
comb_L_arr = []
comb_L_ip_arr = []
comb_median_mass_arr = []
comb_error_arr = []
for i in range(len(stacks_dipole)):   
    x_comb, L_comb, median_value_comb, error_comb = stats.combined_likelihood(x, L_arr_sorted[i], normalize = True, finer_reso = False)
    print(median_value_comb, error_comb) 
    comb_L_arr.append(L_comb)
    M_ip_comb, L_ip_comb, median_value_comb_finer, error_comb_finer = stats.combined_likelihood(x, L_arr_sorted[i], normalize = True, finer_reso = True)
    print(median_value_comb_finer, error_comb_finer) 
    comb_L_ip_arr.append(L_ip_comb)
    comb_median_mass_arr.append(median_value_comb_finer)
    comb_error_arr.append(error_comb_finer)
   
###################################################################################################################################
#%%

# checking dipoles
dipole_checks = [stacks_dipole[0], stacks_dipole[1], stacks_dipole[2]]

_, extent = utils.make_grid([dipole_checks[0].shape[0], reso_arcmin, dipole_checks[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(dipole_checks), 1, figsize=(7,7))
titles = ['Baseline', '+ kSZ Baseline', '+ kSZ']

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

#%%

# checking likelhihood curves
nber_runs_check = 1
start = 0
end = 100
start_finer = 0
end_finer = 1000
mass_int_finer = np.arange(min(mass_int),max(mass_int),0.001) 

_, extent = utils.make_grid([stacks_dipole[0].shape[0], reso_arcmin, stacks_dipole[0].shape[1], reso_arcmin])
fig, axarr = plt.subplots(len(L_arr_sorted), 1, figsize=(7,10))
titles = ['Baseline', '+ kSZ Baseline','+ kSZ']

for i in range(len(axarr)):
    titstr = titles[i]
    axarr[i].set_title(titstr, fontsize = fontsize-6)
    for j in  range(nber_runs_check):
        axarr[i].plot(mass_int[start:end], L_arr_sorted[i][j][start:end], ls = '-', marker = 'x', color = 'blue')
       # axarr[i].plot(mass_int_finer[start_finer:end_finer], L_ip_arr_sorted[i][j][start_finer:end_finer], ls = '-', marker = '', color = 'deepskyblue')   
   # axarr[i].plot(mass_int[start:end], comb_L_arr[i][start:end], ls = '-', marker = 'x', color = 'firebrick')
   # axarr[i].plot(mass_int_finer[start_finer:end_finer], comb_L_ip_arr[i][start_finer:end_finer], ls = '-', marker = '', color = 'red')
    axarr[i].axvline(1.47, color = 'forestgreen', ls = '--')
    axarr[i].set_xlabel(r'$M_{200c}\ $'+'[10M_\sun]', fontsize = fontsize-6)
    #axarr[i].set_ylabel(r'$\text{Normalized}\ $'+'\mathcal{L}', fontsize = fontsize-6)
    axarr[i].tick_params(labelsize = labelsize-6)
plt.show()

#%%

mass_arr = [median_mass_arr_sorted[0], median_mass_arr_sorted[1]]
comb_mass_arr = [comb_median_mass_arr[0], comb_median_mass_arr[1]]
comb_mass_error_arr = [comb_error_arr[0], comb_error_arr[1]]

fig, ax = plt.subplots(figsize=(4,5))
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

ax.set_xlabel(r'$M_{\rm fit}-M_{\rm true}$', fontsize = fontsize)
for i in range(len(xvals)):
    ax.text(xvals[i], yvals[i], titles_arr[i], fontsize = fontsize)
    
#for i in range(len(mass_results_arr)):
#    yval = [yvals[i] for j in range(len(mass_results_arr[i]))]
#    ax.plot(np.round(mass_arr[i], 3)-1.47, yval, color = color_arr[i], marker = markers[i], markersize = 12, ls = '', alpha = 0.25)
for i in range(len(comb_mass_arr)):  
    ax.errorbar((np.round(comb_mass_arr[i], 2)-1.47), yvals[i], xerr = np.round(comb_mass_error_arr[i],2), elinewidth = 4, color = color_arr[i],  marker = markers[i], markersize = 12, ls = '', capsize=5, capthick=3)
plt.show()





















































