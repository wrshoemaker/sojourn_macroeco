from __future__ import division
import config
import os
import sys
import subprocess
import random
import re
import gzip
from collections import Counter
import itertools
import scipy.stats as stats
from scipy.special import digamma, gamma, erf, loggamma, hyperu, polygamma

from scipy import integrate

import numpy
#import phylo_utils
from datetime import datetime


from statsmodels.base.model import GenericLikelihoodModel


numpy.random.seed(123456789)



def estimate_normalization_constant(range_, values_):
    
    # for Py3.11
    norm_factor = integrate.simpson(values_, range_)

    return norm_factor



def log_log_regression(x_, y_, min_x=None):

    if min_x != None:

        x_idx = (x_>=min_x)
        x_ = x_[x_idx]
        y_ = y_[x_idx]

    slope, intercept, r_value, p_value, std_err = stats.linregress(numpy.log10(x_), numpy.log10(y_))

    return slope, intercept, r_value, p_value, std_err


def estimate_sojourn_vs_constant_relationship(run_length, mean_run_deviation):

    norm_constant = []

    for run_length_i_idx, run_length_i in enumerate(run_length):

        mean_run_deviation_i = numpy.asarray(mean_run_deviation[run_length_i_idx])
        s_range_i = numpy.linspace(0, 1, num=run_length_i, endpoint=True)
        norm_constant.append(estimate_normalization_constant(s_range_i, mean_run_deviation_i))

    
    norm_constant = numpy.asarray(norm_constant)
    slope, intercept = log_log_regression(run_length, norm_constant, min_x=10)

    return slope, intercept, norm_constant





def expected_value_log_gamma(x_bar, cv):

    beta = (cv)**(-2)

    return digamma(beta) - numpy.log(beta/x_bar) 




def expected_value_square_root_gamma(x_bar, cv):

    beta = (1/cv)**2

    return (gamma(beta+0.5)/gamma(beta))*numpy.sqrt(x_bar/beta)


def _ll_gamma_sampling(n, N, x_mean, x_std):
    # n = exogenous
    # N = endogenous

    beta = (x_mean/x_std)**2
    ll =  loggamma(beta + n) - loggamma(beta) - loggamma(n+1) + n*(numpy.log(N*x_mean) - numpy.log(beta + N*x_mean)) + beta*(numpy.log(beta) - numpy.log(beta + N*x_mean))

    return ll


class mle_gamma_sampling(GenericLikelihoodModel):
    
    def __init__(self, endog, exog, **kwds):
        super(mle_gamma_sampling, self).__init__(endog, exog, **kwds)

    def nloglikeobs(self, params):
        x_mean = params[0]
        x_std = params[1]
        ll = -1*_ll_gamma_sampling(self.exog.flatten(), self.endog, x_mean, x_std)
        return ll

    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method="bfgs", **kwds):

        #print(type(start_params).__module__, numpy.__name__ )
        #if (type(start_params).__module__ == numpy.__name__ ) == False:

        if type(start_params) == type(None):
            x_mean_start = 0.001
            x_std_start = 0.0001
            start_params = numpy.array([x_mean_start, x_std_start])

        return super(mle_gamma_sampling, self).fit(start_params=start_params, maxiter=maxiter, method = method, maxfun=maxfun, **kwds)









def test_mle():

    n = numpy.asarray([3.000e+00, 1.000e+00, 0.000e+00, 9.000e+00, 6.000e+00, 4.000e+00,
                    5.000e+00, 0.000e+00, 1.000e+00, 1.000e+00, 2.000e+00, 1.000e+00,
                    2.000e+00, 3.000e+00, 1.600e+01, 0.000e+00, 0.000e+00, 7.000e+00,
                    0.000e+00, 1.000e+00, 6.000e+00, 1.000e+00, 1.036e+03, 6.400e+01])


    N = numpy.asarray([8344, 7107, 8644, 8226, 7104, 7213, 7753, 5525, 8556, 6594, 8805, 8629,
                        8293, 7596, 5507, 3397, 7961, 6312, 7572, 6432, 8435, 7746, 8650, 8557])
        
    mu_start = numpy.mean(n/N)
    sigma_start = numpy.std(n/N)
    start_params = numpy.asarray([mu_start, sigma_start])

    gamma_sampling_model = mle_gamma_sampling(N, n)
    gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=start_params, bounds= [(0.000001,1), (0.00001,100)], full_output=False, disp=False)
    #gamma_sampling_model_ll = gamma_sampling_model.loglike(gamma_sampling_result.params)

    #print(gamma_sampling_result.mle_retvals)
    #print(mu_start, sigma_start)
    #print(gamma_sampling_result.params)




def autocorrelation_by_days(data, days, min_n_autocorr_values=5):

    delay_days_range = numpy.arange(1, max(days) - min(days) - min_n_autocorr_values)
    # Rescale data by mean
    data = data - numpy.mean(data)

    delay_days_all = []
    autocorr_all = []

    for delay_days in delay_days_range:

        autocorr = []
        
        # Loop through all possible lags
        for i in range(len(data) - delay_days):
            current_day = days[i]
            lagged_day = days[i + delay_days]
            
            # Check if the difference in days equals delay_days
            if lagged_day - current_day == delay_days:
                current_value = data[i]
                lagged_value = data[i + delay_days]
                # Save product of current and lagged value
                autocorr.append((current_value) * (lagged_value))

        # Sufficient number of datapoints to calcualte autocorrelation
        if len(autocorr) >= min_n_autocorr_values:
            # Normalize covariance by sum of covariance terms divided by # of covariance terms
            # and the variance  
            delay_days_all.append(delay_days)  
            autocorr_all.append(numpy.sum(autocorr) / (len(autocorr) * numpy.var(data)))


    delay_days_all = numpy.asarray(delay_days_all)
    autocorr_all = numpy.asarray(autocorr_all)

    delay_days_all = numpy.insert(delay_days_all, 0, 0, axis=0)
    autocorr_all = numpy.insert(autocorr_all, 0, 1, axis=0)


    return delay_days_all, autocorr_all



def autocorrelation(x):
    # Normalize the input array
    x = numpy.asarray(x)
    x = x - numpy.mean(x)
    
    # Compute the autocorrelation for all lags
    result = numpy.correlate(x, x, mode='full')
    print(len(result))
    
    # Normalize the result by the value at lag 0
    autocorr = result[result.size // 2:]  # Take the second half (positive lags)
    autocorr /= autocorr[0]  # Normalize to have value 1 at lag 0
    
    return autocorr


if __name__ == "__main__":

    test_mle()