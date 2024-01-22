#%%
# importing relevant modules
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
#from matplotlib import rc;rc('text', usetex=True);rc('font', weight='bold');matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
rcParams['font.family'] = 'serif'
#rc('text.latex',preamble=r'\\usepackage{/Users/kevinlevy/Documents/cmb_cluster_lensing/apjfonts}')
color_arr = ['firebrick', 'deepskyblue', 'forestgreen', 'orange', 'purple', 'black', 'grey', 'lime']
import cosmo
import mockobs
import utils
import lensing

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


#####################################################################################################################################################################
#%%

# getting kSZ maps
print('Getting kSZ maps ...')
fname = '/Volumes/Extreme_SSD/cluster_lensing/sims/ksz_tsz/mdpl2_cutouts_for_ksz_clus_detection_M8.5e+13to1.15e+14_z0.65to0.75_32995haloes_boxsize60.0am_dx0.5am.npy'
ksz_cutouts = np.load(fname)

nx_cutout, ny_cutout = ksz_cutouts[0].shape[0], ksz_cutouts[0].shape[1]
s, e = int((nx-nx_cutout)/2), int((ny+ny_cutout)/2)
 
plt.imshow(np.mean(ksz_cutouts[:2500], axis = 0)[50:70, 50:70])
plt.colorbar()
plt.show()


#%%
    

def get_angle_and_magnitude(signal, dx):
    gradient_map = np.nan_to_num(np.gradient(signal, dx))
    gradient_xmap, gradient_ymap = gradient_map[1], gradient_map[0]
    angle_map = np.degrees(np.arctan2(gradient_ymap, gradient_xmap))
    magnitude_map = np.hypot(gradient_xmap, gradient_ymap) 
    return angle_map, magnitude_map    
    
    
def get_aligned_map(map_params, image, image_noiseless = None, cutout_size_for_grad_est_am = 6, l_cut = 2000, l = None, cl = None, cl_noise = None):  

    if image_noiseless is None:
        image_noiseless = np.copy(image)
        
    if cl_noise is None:
        l = np.arange(10001)
        cl = np.ones(max(l)+1)
        cl_noise = np.zeros(max(l)+1)
    
    _, dx, _, _ = map_params
    wiener_filter = utils.wiener_filter(cl, cl_noise)
    low_pass_filter = utils.low_pass_filter(l, l_cut)
    gradient_map = utils.convolve(image, l, wiener_filter * low_pass_filter, map_params) 
    gradient_cutout = utils.central_cutout(gradient_map, cutout_size_for_grad_est_am, map_params)
    angle, magnitude = get_angle_and_magnitude(gradient_cutout, dx)
    angle, magnitude_weight = np.median(angle), np.median(magnitude) 
    image_aligned = utils.rotate(image_noiseless, angle)
    
    return image_aligned, magnitude_weight, angle 


lensing_reso = 1
map_params_lensing = [map_params[0]*lensing_reso, map_params[1]/lensing_reso, map_params[2]*lensing_reso, map_params[3]/lensing_reso]
print(map_params_lensing)
sim = mockobs.cmb_mock_data(map_params_lensing, l, cl, lensing_reso = lensing_reso)
noise_map = utils.make_gaussian_realization(l, nl, map_params) 
sim_background = utils.downsample_map(sim, lensing_reso)
sim_background = utils.convolve(sim_background, l, np.sqrt(bl), mapparams = map_params)   
sim_background_noise = np.copy(sim_background)     
sim_background_noise = sim_background_noise + noise_map 
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
map_aligned_bg, weight_gradient_bg, angle_bg = get_aligned_map(map_params, sim_background_noise, image_noiseless = sim_background, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
print(weight_gradient_bg, angle_bg)
#cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
#map_aligned_bg, weight_gradient_bg, angle_bg = get_aligned_map(map_params, sim_background_noise, image_noiseless = sim_background, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
#print(weight_gradient_bg, angle_bg)
            
            
x_shift, y_shift = np.random.normal(loc=0.0, scale = 0/(2**0.5)), np.random.normal(loc=0.0, scale = 0/(2**0.5)) 
centroid_shift = [x_shift, y_shift]
grid, _ = utils.make_grid(map_params_lensing, grid_shift = centroid_shift) 
theta =  np.hypot(grid[0], grid[1]) 
c200c = cosmo.concentration_parameter(2e14, z, 0.674)
kappa_map = lensing.NFW_convergence(2e14, c200c, z, 1100, theta, dim = 2)
alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params_lensing)                
sim_lensed = lensing.lens_map(sim, alpha_vec, map_params_lensing) 
sim_lensed = utils.downsample_map(sim_lensed, lensing_reso)  
sim_lensed_noise = np.copy(sim_lensed)
sim_lensed_noise_ksz = np.copy(sim_lensed)
rand_sel = random.randint(0, len(ksz_cutouts)-1)
sim_lensed_noise_ksz = sim_lensed_noise_ksz + ksz_cutouts[rand_sel]
sim_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), mapparams = map_params)
sim_lensed_noise = utils.convolve(sim_lensed_noise, l, np.sqrt(bl), mapparams = map_params)
sim_lensed_noise_ksz = utils.convolve(sim_lensed_noise_ksz, l, np.sqrt(bl), mapparams = map_params)
sim_lensed_noise = sim_lensed_noise + noise_map 
sim_lensed_noise_ksz = sim_lensed_noise_ksz + noise_map 
cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l, beam_fwhm)
map_aligned, weight_gradient, angle = get_aligned_map(map_params, sim_lensed_noise, image_noiseless = sim_lensed, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
map_aligned_ksz, weight_gradient_ksz, angle_ksz = get_aligned_map(map_params, sim_lensed_noise_ksz, image_noiseless = sim_lensed, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
print(weight_gradient, angle)
print(weight_gradient_ksz, angle_ksz )
#cl_noise = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)
#map_aligned, weight_gradient, angle = get_aligned_map(map_params, sim_lensed_noise, image_noiseless = sim_lensed, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
#map_aligned_ksz, weight_gradient_ksz, angle_ksz = get_aligned_map(map_params, sim_lensed_noise_ksz, image_noiseless = sim_lensed, cutout_size_for_grad_est_am = cutout_size_for_grad_est_am, l = l, cl = cl, cl_noise = cl_noise)
#print(weight_gradient, angle)
#print(weight_gradient_ksz, angle_ksz) 



