# importing relevant modules
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy as sp
import scipy.optimize as optimize


#################################################################################################################################


def jackknife_resampling(total, no_of_samples):
    
    each_split_should_contain=int(total * 1./no_of_samples)
    fullarr=np.arange(total)
    inds_to_pick=np.arange(len(fullarr))
    jk_samples=[]
    for n in range(no_of_samples):
        inds=np.random.choice(inds_to_pick, size=each_split_should_contain, replace=0)
        inds_to_delete=np.where (np.in1d(inds_to_pick, inds) == True)[0]
        inds_to_pick=np.delete(inds_to_pick, inds_to_delete)
        tmp=np.in1d(fullarr, inds)
        non_inds=np.where(tmp == False)[0]
        jk_samples.append( (non_inds) )
        
    return np.asarray(jk_samples)


#################################################################################################################################


def covariance_and_correlation_matrix(sample, jackknife = False):
    
    matrix = np.concatenate(sample)
    N = len(sample)
    if sample[0].ndim == 1:
        nber_data_points = sample[0].shape[0]
    else:
        nber_data_points = sample[0].shape[0]*sample[0].shape[1]
    matrix = matrix.flatten().reshape(N, nber_data_points)
    covariance_matrix = np.cov(matrix, rowvar=False) 
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    
    if jackknife is True:
        covariance_matrix *= ((N-1)/N)*(N-1)
    
    return covariance_matrix, correlation_matrix


#################################################################################################################################


def get_lnlikelihood(data, model, cov):
    """
    function to calculate the likelihood given data, model, covariance matrix
    """
    cov=np.mat(cov)
    cov_inv=sp.linalg.pinv(cov)

    d=data.flatten()
    m=model.flatten()
    d=d-m
   # diff = (data - model).flatten()
   # chi_2 = np.dot(diff.T, np.dot(np.linalg.pinv(cov), diff))
   # logLval =  -0.5*chi_2
    logLval= -0.5 * np.asarray( np.dot(d.T, np.dot( cov_inv, d ))).squeeze()

    return logLval



def lnlike_to_like(M, lnlike):

    lnlike=lnlike - max(lnlike)

    '''
    if intrp_type == 1:
        deltaM=np.diff(M)[0]
        M_ip=np.arange(min(M),max(M),deltaM/100.)
        lnlike_ip=np.interp(M_ip, M, lnlike)
        M=np.copy(M_ip)
        lnlike=np.copy(lnlike_ip)
    '''

    delta_chisq=max(lnlike) - lnlike[0]
    snr=np.sqrt(2 * delta_chisq)
    L=np.exp(lnlike)
    L/=max(L)
    recov_mass=M[np.argmax(L)]
    
    return M, L, recov_mass, snr


def random_sampler(x, y, howmanysamples = 1000000, burn_in = 5000):

    norm = integrate.simps(y, x) #area under curve for norm
    y = y/norm #normalise dn/dM here

    cdf = np.asarray([integrate.simps(y[:i+1], x[:i+1]) for i in range(len(x))])
    cdf_inv = interpolate.interp1d(cdf, x)

    random_sample = cdf_inv(np.random.rand(howmanysamples))

    return random_sample[burn_in:]  


def get_width_from_sampling(x, likelihood_curve, nsamples = 1000000, burn_in = 5000):#, sigma_value = [1.]):
    randoms = random_sampler(x, likelihood_curve, howmanysamples=nsamples, burn_in = burn_in)
    median_mass = np.percentile(randoms, 50.)
    low_err = median_mass - np.percentile(randoms, 16.)
    high_err = np.percentile(randoms, 84.) - median_mass

    return median_mass, (high_err+low_err)/2


def fitting_func_gaussian(p, p0, X, DATA = None, return_fit = 0):
    fitfunc = lambda p, x: p[1]*(np.exp(-(x-p[2])**2/(2*p[3]**2)))
    if not return_fit:
        return fitfunc(p, X) - DATA
    else:
        return fitfunc(p, X)

def likelihood_finer_reso(M, L, delta=0.001):
    M_ip = np.arange(min(M),max(M),delta) 
    best_fit = M[np.argmax(L)]
    gau_width = abs(best_fit - M[np.argmin(abs(L))])/2.35 * 2.
    p0 = [0.,np.max(L),best_fit,gau_width]
    p1, success = optimize.leastsq(fitting_func_gaussian, p0, args=(p0, M, L))
    L_ip = fitting_func_gaussian(p1, p1, M_ip, return_fit = 1)
    return M_ip, L_ip

def run_ml(data, models, cov, param_int, nsamples = 1000000, burn_in = 5000, finer_reso = True):
    lnlike_arr = []
    for model in models:
        lnlike_val = get_lnlikelihood(data,  model, cov)
        lnlike_arr.append(lnlike_val)
    x, L, recov_mass, snr = lnlike_to_like(param_int, lnlike_arr)
    if finer_reso is True:
        x, L, = likelihood_finer_reso(x, L, delta=0.001)
        L /= max(L)
    median_value, error = get_width_from_sampling(x, L, nsamples = nsamples, burn_in = burn_in)
    
    return x, L, median_value, error


#################################################################################################################################


def combined_likelihood(x, likelihood_arr, normalize = True, finer_reso = True):
    comb_lk = np.ones(len(likelihood_arr[0]))
    for i in range(len(likelihood_arr)):
        comb_lk += np.log(likelihood_arr[i])
    comb_lk = np.exp(comb_lk)
    if normalize is True:
        comb_lk = comb_lk/max(comb_lk) 
    if finer_reso is True:
        x, comb_lk = likelihood_finer_reso(x, comb_lk, delta=0.001)
        comb_lk /= max(comb_lk) 
    median_value, error = get_width_from_sampling(x, comb_lk, nsamples = 1000000, burn_in = 5000)
    return x, comb_lk, median_value, error


def signal_to_noise(pdf):
    lnpdf = np.log(pdf)
    delta_chisq = 2*(max(lnpdf) - lnpdf[0])
    snr = np.sqrt(delta_chisq)
    return snr