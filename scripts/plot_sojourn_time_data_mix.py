import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
from collections import Counter
import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma
from scipy import integrate

#import functools
#import operator

import data_utils
import plot_utils
import stats_utils
import simulation_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors, colorbar

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


def make_null_gamma_sojourn_time_dist(max_sojourn_time=1000):

    n_obs_per_dist = []
    days_run_lengths_all = []

    sojourn_time_range = numpy.arange(1, max_sojourn_time+1)

    sojourn_time_pdf_all = []
    for dataset in data_utils.dataset_all:
    
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            for asv in  mle_dict[dataset][host].keys():
                
                days_run_lengths = mle_dict[dataset][host][asv]['days_run_lengths']
                

                x_cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_mean']
                x_beta = (1/x_cv)**3
                
                expected_log_x = stats_utils.expected_value_log_gamma(1, x_cv)

                #print(expected_log_x)

                cdf_value = stats.loggamma.cdf(expected_log_x, c=x_beta, scale=1)

                #z = (cdf_value/(1-cdf_value)) + ((1-cdf_value)/cdf_value)
                # normalization constant
                z = (cdf_value/(1-cdf_value))

                #sojourn_time_pdf = ((cdf_value**sojourn_time_range) + ((1-cdf_value)**sojourn_time_range))/z
                sojourn_time_pdf = ((cdf_value**sojourn_time_range))/z
                sojourn_time_pdf_all.append(sojourn_time_pdf)
                n_obs_per_dist.append(len(days_run_lengths))
                days_run_lengths_all.extend(days_run_lengths)




    # calculate mixture
    sojourn_time_pdf_all = numpy.stack(sojourn_time_pdf_all, axis=0)
    n_obs_per_dist = numpy.asarray(n_obs_per_dist)
    weights_per_dist = n_obs_per_dist/sum(n_obs_per_dist)
    mixture_dist = numpy.sum(sojourn_time_pdf_all * weights_per_dist[:, numpy.newaxis], axis=0)

    # get empirical distribution
    days_run_lengths_dict = dict(Counter(days_run_lengths_all))
    sojourn_data_range = numpy.sort(list(days_run_lengths_dict.keys()))
    #sojourn_data_range.sort()
    sojourn_data_pdf = numpy.asarray([days_run_lengths_dict[s] for s in sojourn_data_range])
    sojourn_data_pdf = sojourn_data_pdf/sum(sojourn_data_pdf)

    # match empirical range to prediction
    cutoff_idx = numpy.where(sojourn_time_range == max(sojourn_data_range))[0][0]
    sojourn_null_range = sojourn_time_range[:cutoff_idx]
    sojourn_null_pdf = mixture_dist[:cutoff_idx]


    return sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf
    



#colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}


fig, ax = plt.subplots(figsize=(4,4))

sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = make_null_gamma_sojourn_time_dist()

ax.plot(sojourn_data_range, sojourn_data_pdf, c='#87CEEB', lw=2, ls='-', label='Data')
ax.plot(sojourn_null_range, sojourn_null_pdf, c='k', lw=2, ls=':', label=r'$\tau \ll \delta t $' + ' (gamma)')

ax.set_xlim([1, max(sojourn_data_range)])
ax.set_ylim([min(sojourn_data_pdf), max([max(sojourn_data_pdf), max(sojourn_null_pdf)])])

ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)



ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=12)
ax.set_ylabel("Probability density, " + r'$P(\mathcal{T})$', fontsize=12)

ax.legend(loc='upper right', fontsize=10)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_data_mix.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()



