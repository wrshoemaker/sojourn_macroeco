
import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma
from scipy import integrate

#import functools
#import operator

import data_utils
import plot_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import stats_utils
import simulation_utils

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))




dataset = 'david_et_al'

host_all = list(mle_dict[dataset].keys())
host_all.sort()

tau_all = []
sigma_all = []


sigma__ = []
for host_idx, host in enumerate(host_all):

    for key, value in mle_dict[dataset][host].items():
        
        x_mean = value['x_mean']
        x_std = value['x_std']
        x_cv = x_std/x_mean
        rel_abundance = numpy.asarray(value['rel_abundance'])
        ln_rescaled_rel_abundance = numpy.log(rel_abundance/x_mean)
        days = numpy.asarray(value['days'])
        k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
        sigma__.append(sigma)


        delay_days_all, autocorr_all = stats_utils.autocorrelation_by_days(ln_rescaled_rel_abundance, days, min_n_autocorr_values=5)

        filter_idx = (autocorr_all > 0) & (autocorr_all < 1.0)

        if sum(filter_idx) < 4:
            continue

        delay_days_all_filter = delay_days_all[filter_idx]
        autocorr_all_filter = autocorr_all[filter_idx]
        log_autocorr_all_filter = numpy.log(autocorr_all_filter)

        # fit slope with intercept of zero
        slope = numpy.dot(delay_days_all_filter, log_autocorr_all_filter) / numpy.dot(delay_days_all_filter, delay_days_all_filter)
        # backcalculate tau



        tau = -1*(1 - (sigma/2))/slope

        tau_all.append(tau)
        sigma_all.append(sigma)



#print(sigma, tau)

print(numpy.corrcoef(tau_all, sigma_all)[0,1])



sigma__ = numpy.asarray(sigma__)
rescaled_sigma = (((2/sigma__) - 1)**(-0.5) ) * numpy.sqrt(8/numpy.pi)

print( numpy.mean(rescaled_sigma))
#print(min(sigma__), max(sigma__))

