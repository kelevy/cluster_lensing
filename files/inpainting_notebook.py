#%%
print('Importing relevant modules ...')
import warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import rcParams
color_arr = ['olivedrab', 'steelblue', 'darkgoldenrod', 'indianred', 'darkorchid']
import cosmo
import mockobs
import inpaint
rcParams['figure.dpi'] = 150
rcParams["figure.facecolor"] = 'white'
cmap = cm.RdYlBu


#params or supply a params file
dx, dy = 0.5, 0.5
reso_arcmin=0.5
nx, ny = 120, 120
x1,x2 = -nx/2. * dx, nx/2. * dx
map_params = [nx, dx, ny, dy]
verbose = 0
# for plotting
fontsize = 20
labelsize = 17

#beam and noise levels
noiseval_white = 2.0 #uK-arcmin
beam_fwhm = 1.0 #arcmins

#for inpainting
noofsims = 500
mask_radius_inner = 4.0 #arcmins
mask_radius_outer = 16.0 #arcmins
mask_inner = 0  #If 1, the inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
#get ra, dec or map-pixel grid
ra = np.linspace(x1,x2, nx) #arcmins
dec = np.linspace(x1,x2, nx) #arcmins
ra_grid, dec_grid = np.meshgrid(ra,dec)



#get beam and noise
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
bl = mockobs.beam_power_spectrum(beam_fwhm, l)
nl = mockobs.instrumental_noise_power_spectrum(noiseval_white, l)

#%%
sim_unlensed = mockobs.cmb_mock_data(map_params, l, cl, nber_obs = 1)
import utils
import lensing
M_input = 8e14
z=0.7
grid, _ = utils.make_grid(map_params) 
theta =  np.hypot(grid[0], grid[1]) 
c200c = cosmo.concentration_parameter(M_input, z, 0.674)
kappa_map = lensing.NFW_convergence(M_input, c200c, z, 1100, theta, dim = 2)
alpha_vec = lensing.deflection_from_convergence(kappa_map, map_params)   
sims_lensed = np.copy(sim_unlensed)
sim_lensed = lensing.lens_map(sim_unlensed, alpha_vec, map_params) 

sim_white_noise = utils.make_gaussian_realization(l, nl, map_params) 
mockob_unlensed = utils.convolve(sim_unlensed, l, np.sqrt(bl), mapparams = map_params) + sim_white_noise
mockob_lensed = utils.convolve(sim_lensed, l, np.sqrt(bl), mapparams = map_params) + sim_white_noise
#%%
inpainted_map = inpaint.inpainting(mockob_lensed, ra_grid, dec_grid, map_params, l, cl, bl, nl, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff = 1, mask_inner = mask_inner)

#%%

reso_arcmin=dx
_, extent = utils.make_grid([mockob_unlensed[54:66, 54:66].shape[0], reso_arcmin, mockob_unlensed[54:66, 54:66].shape[1], reso_arcmin])

ax=plt.subplot(234)
plt.imshow(mockob_unlensed[54:66, 54:66], extent =extent, cmap = cmap)
plt.colorbar(); 
plt.title(r'unlensed map')


ax=plt.subplot(235)
plt.imshow(mockob_lensed[54:66, 54:66], extent = extent, cmap = cmap)
plt.colorbar()
plt.title(r'lensed map')


ax=plt.subplot(236)
plt.imshow(inpainted_map[54:66, 54:66], extent = extent, cmap = cmap)
plt.colorbar()
plt.title(r'inpainted map')



#residual_flatten = (residual[s:e, s:e].flatten())*1e3
#minval, maxval, delta = -1., 1., 0.1
#binbin = np.arange(minval, maxval, delta)
#plt.subplot(236)
#plt.hist(residual_flatten, bins = binbin, histtype = 'step', color = 'black')
#plt.title(r'Residual')
#plt.xlabel(r'Residual [nK]')
#plt.show()
