#%%
print('Importing relevant modules ...')
# general modules
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
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
color_arr = ['olivedrab', 'steelblue', 'darkgoldenrod']

# own modules
import cosmo
import lensing
import mockobs
import lensing_estimator
import stats
import utils


####################################################################################################################################
#%%


print('Defining relevant parameters ...')
nber_clus = 10000
nber_rand = 10000
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
M_input = 2e14


# for plotting
xmin, xmax = -nx*reso_arcmin/2, nx*reso_arcmin/2
ymin, ymax = -ny*reso_arcmin/2, ny*reso_arcmin/2 
extent_arcmin = [xmin, xmax, ymin, ymax]
extent_degrees = np.asarray(extent_arcmin)/60
fontsize = 34
labelsize = 30


###################################################################################################################################
#%%


sims_unlensed = mockobs.cmb_mock_data(map_params, l, cl, cluster = None, centroid_shift_value = 0, nber_ch = 1, cluster_corr_cutouts = None, cl_extragal = None, bl = bl, nl = nl, nber_obs = nber_rand, lensing_reso = 1)
sims_lensed = mockobs.cmb_mock_data(map_params, l, cl, cluster = [M_input, cosmo.concentration_parameter(M_input, z, 0.674), z], centroid_shift_value = 0, nber_ch = 1, cluster_corr_cutouts = None, cl_extragal = None, bl = bl, nl = nl, nber_obs = nber_clus, lensing_reso = 1)

angle_arr_unlensed = []
for sim_unlensed in sims_unlensed:
    image_aligned, magnitude_weight, angle  = lensing_estimator.get_aligned_map_0(map_params, sim_unlensed, image_noiseless = None,
                                                                                  cutout_size_for_grad_est_am = 6, l_cut = 2000, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, inpaint_map = False, bl = bl, nl = nl, sigma_dic = None,
                                                                                  mask_radius_inner = None, mask_radius_outer = None, noofsims = 1000,  mask_inner = 0)
    angle_arr_unlensed.append(angle)
    
angle_arr_lensed = []
for sim_lensed in sims_lensed:
    image_aligned, magnitude_weight, angle  = lensing_estimator.get_aligned_map_0(map_params, sim_lensed, image_noiseless = None,
                                                                                  cutout_size_for_grad_est_am = 6, l_cut = 2000, l = l, cl = cl, 
                                                                                  cl_noise = cl_noise, inpaint_map = False, bl = bl, nl = nl, sigma_dic = None,
                                                                                  mask_radius_inner = None, mask_radius_outer = None, noofsims = 1000,  mask_inner = 0)
    angle_arr_lensed.append(angle)
    
###################################################################################################################################
#%%

plt.hist(angle_arr_unlensed, bins=100)
plt.show()
plt.subplots()
plt.hist(angle_arr_lensed, bins=100)
plt.show()
