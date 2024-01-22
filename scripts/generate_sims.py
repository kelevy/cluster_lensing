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

maps_rand = mockobs.cmb_mock_data(map_params, l, cl, bl = bl, nl = nl, nber_obs = nber_rand)
np.save('sims/unlensed_mockobs/60arcmin_1.0arcmin_2.0uKarcmin_50000sims.npy', maps_rand)
del maps_rand

maps_clus = mockobs.cmb_mock_data(map_params, l, cl, cluster = [mass, cosmo.concentration_parameter(mass, z, 0.674), z], bl = bl, nl = nl, nber_obs = nber_clus)
np.save('sims/lensed_mockobs/60arcmin_2e14Msolar_1.0arcmin_2.0uKarcmin_2500sims.npy', maps_clus)
del maps_clus
