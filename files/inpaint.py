import numpy as np, mockobs, utils
import scipy as sc
from pylab import *


#################################################################################
#################################################################################
#################################################################################

def calccov(sim_mat, noofsims, npixels):
    
    m = sim_mat.flatten().reshape(noofsims,npixels)
    m = np.mat( m ).T
    mt = m.T

    cov = (m * mt) / (noofsims)# - 1)
    return cov

#################################################################################
def get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer, square = 0, in_arcmins = 1):

    if not in_arcmins:
        ra_grid = ra_grid * 60.
        dec_grid = dec_grid * 60.

    if not square:
        radius = np.sqrt( (ra_grid**2. + dec_grid**2.) )
        inds_inner = np.where((radius<=mask_radius_inner))
        inds_outer = np.where((radius>mask_radius_inner) & (radius<=mask_radius_outer) )
    else:
        inds_inner = np.where( (abs(ra_grid)<=mask_radius_inner) & (abs(dec_grid)<=mask_radius_inner) )
        ##inds_outer = np.where( (abs(ra_grid_arcmins)>mask_radius_inner) & (abs(ra_grid_arcmins)<=mask_radius_outer) & (abs(dec_grid_arcmins)>mask_radius_inner) & (abs(dec_grid_arcmins)<=mask_radius_outer))
        inds_outer = np.where( (abs(ra_grid)<=mask_radius_outer) & (abs(dec_grid)<=mask_radius_outer) & ( (abs(ra_grid)>mask_radius_inner) | (abs(dec_grid)>mask_radius_inner) ) )

    return inds_inner, inds_outer

#################################################################################
def get_covariance(ra_grid, dec_grid, mapparams, el, cl, bl, nl, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff = 1, maxel_for_grad_filter = None):


    ############################################################
    #get the low pass filter
    if low_pass_cutoff:
        assert mask_radius_inner is not None
        if maxel_for_grad_filter is None:
            maxel_for_grad_filter = int( 3.14/np.radians(mask_radius_inner/60.) )
        lpf = utils.get_lpf_hpf(mapparams, maxel_for_grad_filter, filter_type = 0)

    ############################################################

    #get the sims for covariance calculation
    sims_for_covariance = []
    for n in range(noofsims):

        #cmb sim and beam, for CMB include the transfer function and beam
        sim_map = mockobs.cmb_mock_data(mapparams, el, cl, bl = bl, nl = nl) # flatsky.make_gaussian_realisation(mapparams, el, cl_dic['TT'], bl = bl)

        ############################################################
        #lpf the map
        if low_pass_cutoff:
            sim_map = np.fft.ifft2( np.fft.fft2(sim_map) * lpf ).real
        #imshow(sim_map);colorbar(); show(); sys.exit()

        sims_for_covariance.append( sim_map )
    sims_for_covariance = np.asarray( sims_for_covariance)

    ############################################################
    #get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    ############################################################
    #get the pixel values in the inner and outer regions
    t1_for_cov = sims_for_covariance[:,inds_inner[0], inds_inner[1]]
    t2_for_cov = sims_for_covariance[:,inds_outer[0], inds_outer[1]]

    ############################################################
    #get the covariance now
    npixels_t1 = t1_for_cov.shape[1]
    npixels_t2 = t2_for_cov.shape[1]

    t1t2_for_cov = np.concatenate( (t1_for_cov,t2_for_cov), axis = 1 )
    npixels_t1t2 = t1t2_for_cov.shape[1]
    t1t2_cov = calccov(t1t2_for_cov, noofsims, npixels_t1t2)


    ############################################################
    #https://arxiv.org/pdf/1301.4145.pdf
    ##Eq. 32
    #sigma_11 = t1t2_cov[:npixels_t1, : npixels_t1] 
    sigma_22 = t1t2_cov[npixels_t1:,npixels_t1:]
    sigma_12 = t1t2_cov[:npixels_t1,npixels_t1:]
    #sigma_21 = t1t2_cov[npixels_t1:,:npixels_t1]


    sigma_22_inv = sc.linalg.pinv(sigma_22)
    sigma_dic = {}
    ##sigma_dic['sigma_11'] = sigma_11
    #sigma_dic['sigma_11_inv'] = sigma_11_inv
    ##sigma_dic['sigma_22'] = sigma_22
    sigma_dic['sigma_22_inv'] = sigma_22_inv
    sigma_dic['sigma_12'] = sigma_12
    ##sigma_dic['sigma_21'] = sigma_21

    return sigma_dic

#################################################################################

def inpainting(img, ra_grid, dec_grid, mapparams, el, cl, bl, nl, noofsims, mask_radius_inner, 
               mask_radius_outer, low_pass_cutoff = 1, intrp_r1_before_lpf = 0, mask_inner = 0,
               maxel_for_grad_filter=None, use_original=False,
               sigma_dic = None):

    """
    mask_inner = 1: The inner region is masked before the LPF. Might be useful in the presence of bright SZ signal at the centre.
    """

    ############################################################
    #get covariance
    if sigma_dic is None:
        sigma_dic = get_covariance(ra_grid, dec_grid, mapparams, el, cl, bl, nl, noofsims, mask_radius_inner, mask_radius_outer, low_pass_cutoff = low_pass_cutoff)
        
    sigma_12 = sigma_dic['sigma_12']
    sigma_22_inv = sigma_dic['sigma_22_inv']

    ############################################################
    #get the low pass filter
    if low_pass_cutoff:
        assert mask_radius_inner is not None
        if maxel_for_grad_filter is None:
            maxel_for_grad_filter = int( 3.14/np.radians(mask_radius_inner/60.) )
        lpf = utils.get_lpf_hpf(mapparams, maxel_for_grad_filter, filter_type = 0)

    ############################################################
    #get the inner and outer pixel indices
    inds_inner, inds_outer = get_mask_indices(ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)

    ############################################################
    map_to_inpaint = np.copy(img)
    original_map = map_to_inpaint.copy()
    ############################################################
    #mask the inner region before LPF if required - might be useful in the presence of bright SZ signal at the centre
    if mask_inner and low_pass_cutoff: #otherwise t2 will see the HPF artefact
        print('\n\n\t\tnot yet implemented\n\n')
        sys.exit()
        mask = masking_for_filtering(ra_grid, dec_grid, simmaps, mask_radius = mask_radius_inner - 2.)
        map_to_inpaint = map_to_inpaint * mask

    ############################################################
    if intrp_r1_before_lpf:
        map_to_inpaint = inpterpolate_r1_using_r2(map_to_inpaint, ra_grid, dec_grid, mask_radius_inner, mask_radius_outer)
    ############################################################

    #lpf the map
    if low_pass_cutoff:
        map_to_inpaint = np.fft.ifft2( np.fft.fft2(map_to_inpaint) * lpf).real
        ##imshow(map_to_inpaint);colorbar();show();sys.exit()
    ############################################################
    #get the pixel values in the inner and outer regions 
    t2_data = map_to_inpaint[inds_outer[0], inds_outer[1]].flatten()

    ############################################################
    #generate constrained Gaussia CMB realisation now
    # include transfer function for the data map
    # include noise twod for the noise

    constrained_sim_to_inpaint = original_map
    #lpf the map
    if low_pass_cutoff:
        constrained_sim_to_inpaint = np.fft.ifft2( np.fft.fft2(constrained_sim_to_inpaint) * lpf ).real


    ############################################################
    #get the pixel values in the inner and outer regions from the constrained realisation
    t1_tilde = constrained_sim_to_inpaint[inds_inner[0], inds_inner[1]].flatten()
    t2_tilde = constrained_sim_to_inpaint[inds_outer[0], inds_outer[1]].flatten()

    ############################################################
    #get the modified t1 values

    
    
    inpainted_t1 = np.asarray( t1_tilde + np.dot(sigma_12, np.dot(sigma_22_inv, ( t2_data - t2_tilde) ) ) )[0]  ##Eq. 36
    ############################################################
    #create a new inpainted map: copy the old map and replace the t1 region
    # if use_original is True, put the inpainted region in the input map before lpf
    if use_original:
        map_to_inpaint = original_map
    inpainted_map = np.copy(map_to_inpaint)

    #inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1
    #split inpainted T(or /Q/U) vector in nx1(or 3) array
    npixels_t1 = int(len(t1_tilde))
    inpainted_t1_tqu_split = inpainted_t1.reshape(1, npixels_t1 )
    inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split

    cmb_inpainted_map = np.copy(map_to_inpaint)*0.
    #cmb_inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1
    cmb_inpainted_map[inds_inner[0], inds_inner[1]] = inpainted_t1_tqu_split

    return inpainted_map

#################################################################################

def masking_for_filtering(ra_grid, dec_grid, mask_radius = 2., taper_radius = 6., in_arcmins = 1):

    import scipy as sc
    import scipy.ndimage as ndimage

    if not in_arcmins:
        ra_grid_arcmins = ra_grid * 60.
        dec_grid_arcmins = dec_grid * 60.

    radius = np.sqrt( (ra_grid_arcmins**2. + dec_grid_arcmins**2.) )

    mask = np.ones( ra_grid_arcmins.shape )
    if (1): 
        inds_to_mask = np.where((radius<=mask_radius)) #2arcmins - fix this for now
        mask[inds_to_mask[0], inds_to_mask[1]] = 0.

    ker=np.hanning(taper_radius)
    ker2d=np.asarray( np.sqrt(np.outer(ker,ker)) )

    mask=ndimage.convolve(mask, ker2d)
    mask/=mask.max()

    return mask
