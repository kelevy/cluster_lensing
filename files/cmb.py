import numpy as np
import cosmo
l = np.arange(10001)
cl = cosmo.cmb_power_spectra(l)['TT']
np.save('/Users/kevinlevy/Documents/cluster_lensing/cmb/planck2018_TTTEEE_lowE_lensing_lensedCls.npy', cl)