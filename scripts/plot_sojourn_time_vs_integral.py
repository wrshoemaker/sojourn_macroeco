from __future__ import division
import config
import os
import sys
import subprocess
import random
import re
import gzip
import pickle
from collections import Counter
import itertools
import numpy
import scipy.stats as stats
from scipy.special import digamma, gamma, erf, loggamma, hyperu, polygamma

from scipy import integrate


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

import data_utils
import plot_utils
import stats_utils
import simulation_utils

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))



#fig, ax = plt.subplots(figsize=(6,4))

fig = plt.figure(figsize = (8.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=2)


ax_time = fig.add_subplot(gs[0, 0])
ax_sigma = fig.add_subplot(gs[0, 1])


run_length_all_all = []
run_sojourn_integral_all_all = []
sigma_all_all = []
for dataset in data_utils.dataset_all:

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host in host_all:

        sys.stderr.write("Analyzing host %s.....\n" % host)

        run_length_all = []
        run_sojourn_integral_all = []
        sigma_all = []

        

        for asv in  mle_dict[dataset][host].keys():
            
            run_dict = mle_dict[dataset][host][asv]['run_dict']

            x_mean = mle_dict[dataset][host][asv]['x_mean']
            x_std = mle_dict[dataset][host][asv]['x_std']
            x_cv = x_std/x_mean

            if run_dict is None:
                continue

            run_sojourn_integral = []

            k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
            
            for run_length, run_sojourn in run_dict.items():

                for run_sojourn_j in run_sojourn:

                    run_sojourn_j = numpy.asarray(run_sojourn_j)

                    run_sojourn_integral_j = integrate.simpson(run_sojourn_j, numpy.linspace(0, 1, num=len(run_sojourn_j), endpoint=True))

                    # run length has units of days
                    run_length_all.append(run_length)
                    run_sojourn_integral_all.append(run_sojourn_integral_j)
                    sigma_all.append(sigma)

                    run_sojourn_integral.append(run_sojourn_integral_j)

                    run_length_all_all.append(run_length)
                    run_sojourn_integral_all_all.append(run_sojourn_integral_j)
                    sigma_all_all.append(sigma)
          

            #if len(run_sojourn_integral) > 2:

            #    pred_mean_run_sojourn_integral = numpy.sqrt(8/numpy.pi) * (((2/sigma) -1)**(-0.5))

            #    print(sigma, pred_mean_run_sojourn_integral, numpy.mean(run_sojourn_integral))

        if len(run_length_all) == 0:
            continue

        sigma_all = numpy.asarray(sigma_all)
        sigma_all = (2/sigma_all) -1
        
        ax_time.scatter(run_length_all, run_sojourn_integral_all, color=plot_utils.host_color_dict[dataset][host], alpha=0.6, s=10)
        ax_sigma.scatter(sigma_all, run_sojourn_integral_all, color=plot_utils.host_color_dict[dataset][host], alpha=0.6, s=10)




# regression...
#sigma_all_all = numpy.asarray(sigma_all_all)
#print(stats_utils.log_log_regression((2/sigma_all_all) -1, run_sojourn_integral_all_all)[0])

slope, intercept, r_value, p_value, std_err = stats_utils.log_log_regression(run_length_all_all, run_sojourn_integral_all_all)

x_log10_range =  numpy.linspace(min(numpy.log10(run_length_all_all)) , max(numpy.log10(run_length_all_all)) , 10000)
y_log10_fit_range = (slope*x_log10_range + intercept)


y_log10_high_tau_range = (0.5*x_log10_range + intercept)
y_log10_low_tau_range = (0*x_log10_range + intercept)


ax_time.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label='Exponent = %.3f' % slope)

ax_time.plot(10**x_log10_range, 10**y_log10_high_tau_range, c='#FF6347', lw=2.5, linestyle=':', zorder=2, label=r'$\tilde{\tau} \gg t, \mathcal{T}-t$')
ax_time.plot(10**x_log10_range, 10**y_log10_low_tau_range, c='#87CEEB', lw=2.5, linestyle=':', zorder=2, label=r'$\tilde{\tau} \ll t, \mathcal{T}-t$')


ax_time.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=12)
ax_time.set_ylabel("Cumulative walk length, " + r'$\int_{0}^{1}  \tilde{y}_{s\mathcal{T}} $', fontsize=12)
ax_time.legend(loc='upper left', fontsize=10)

ax_time.set_ylim([0.08, 50])

ax_time.set_xscale('log', base=10)
ax_time.set_yscale('log', base=10)




####
####
# sigma
sigma_all_all = numpy.asarray(sigma_all_all)
sigma_all_all = (2/sigma_all_all) -1
slope, intercept, r_value, p_value, std_err = stats_utils.log_log_regression(sigma_all_all, run_sojourn_integral_all_all)

x_log10_range =  numpy.linspace(min(numpy.log10(sigma_all_all)) , max(numpy.log10(sigma_all_all)) , 10000)
y_log10_fit_range = (slope*x_log10_range + intercept)

y_log10_high_tau_range = (-0.5*x_log10_range + intercept)
#y_log10_low_tau_range = (0*x_log10_range + intercept)

ax_sigma.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label='Exponent = %.3f' % slope)

ax_sigma.plot(10**x_log10_range, 10**y_log10_high_tau_range, c='#87CEEB', lw=2.5, linestyle=':', zorder=2, label=r'$\tilde{\tau} \ll t, \mathcal{T}-t$')
#ax_sigma.plot(10**x_log10_range, 10**y_log10_low_tau_range, c='#87CEEB', lw=2.5, linestyle=':', zorder=2, label=r'$\tilde{\tau} \ll t, \mathcal{T}-t$')


ax_sigma.set_xlabel('Rescaled sigma', fontsize=12)
ax_sigma.set_ylabel("Cumulative walk length, " + r'$\int_{0}^{1}  \tilde{y}_{s\mathcal{T}} $', fontsize=12)
ax_sigma.legend(loc='lower left', fontsize=10)

#ax_time.set_ylim([0.08, 50])

ax_sigma.set_xscale('log', base=10)
ax_sigma.set_yscale('log', base=10)



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_vs_integral.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()



# 




