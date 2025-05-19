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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import data_utils
import plot_utils
import stats_utils
import simulation_utils

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))




#fig = plt.figure(figsize = (8.5, 4)) #
#fig.subplots_adjust(bottom= 0.15)
#gs = gridspec.GridSpec(nrows=1, ncols=2)


#ax_time = fig.add_subplot(gs[0, 0])
#ax_sigma = fig.add_subplot(gs[0, 1])


run_length_all = []
run_sojourn_integral_all = []
sigma_all = []
dataset_all = []
for dataset in data_utils.dataset_all:

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host in host_all:

        sys.stderr.write("Analyzing host %s.....\n" % host)

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

                    run_length_all.append(run_length)
                    run_sojourn_integral_all.append(run_sojourn_integral_j)
                    sigma_all.append(sigma)
                    dataset_all.append(dataset)
          



run_length_all = numpy.asarray(run_length_all)
run_sojourn_integral_all = numpy.asarray(run_sojourn_integral_all)
sigma_all = numpy.asarray(sigma_all)
#print(numpy.median(sigma_all))
sigma_function_all = (2/sigma_all) -1

dataset_all = numpy.asarray(dataset_all)


fig, ax = plt.subplots(figsize=(5,4))
# intialize_subplot
from matplotlib.patches import Rectangle
rect = Rectangle((0.58,0.5),
                 0.48,0.45,
                 transform=ax.transAxes,
                 facecolor='white',
                 edgecolor='none',
                 linewidth=1.5,
                 zorder=2) 

#ax.add_patch(rect)

ax_sigma = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.58,0.64,0.35,0.35), bbox_transform=ax.transAxes, loc='upper right')
#ax_sigma.set_facecolor('white')
ax_sigma.tick_params(labelleft=False, labelbottom=False, left=False, bottom=False)
ax_sigma.xaxis.set_tick_params(labelsize=6)


for dataset_i in data_utils.dataset_all:

    dataset_idx = numpy.where(dataset_all == dataset_i)[0]
    run_length_i = run_length_all[dataset_idx]
    run_sojourn_integral_i = run_sojourn_integral_all[dataset_idx]
    sigma_function_i = sigma_function_all[dataset_idx]


    ax.scatter(run_length_i, run_sojourn_integral_i, color=plot_utils.dataset_color_dict[dataset_i], alpha=0.6, s=15, label=plot_utils.dataset_name_dict[dataset_i])
    ax_sigma.scatter(sigma_function_i, run_sojourn_integral_i, color=plot_utils.dataset_color_dict[dataset_i], alpha=0.6, s=10)




# regression...
#sigma_all_all = numpy.asarray(sigma_all_all)
slope_time, intercept_time, r_value_time, p_value_time, std_err_time = stats_utils.log_log_regression(run_length_all, run_sojourn_integral_all)
x_log10_range_time =  numpy.linspace(min(numpy.log10(run_length_all)) , max(numpy.log10(run_length_all)) , 10000)
y_log10_time_high_tau = (0.5*x_log10_range_time + intercept_time)
y_log10_time_low_tau = (0*x_log10_range_time + intercept_time)


#ax_time.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label='Exponent = %.3f' % slope)

#ax.plot(10**x_log10_range_time, 10**y_log10_time_high_tau, c='k', lw=3, linestyle='-', zorder=1, label='Prediction: ' + r'$\tau \gg \mathcal{T}$')
ax.plot(10**x_log10_range_time, 10**y_log10_time_low_tau, c='k', lw=3, linestyle='--', zorder=2, label=r'$\tau \ll \mathcal{T}$')


ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=15)
ax.set_ylabel("Sojourn trajectory area, " + r'$\mathcal{A}(\mathcal{T})$', fontsize=15)
ax.legend(loc='upper left', fontsize=10)

ax.set_ylim([0.05, 60])

#min(numpy.log10(run_sojourn_integral_all))*
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

####
####

slope_sigma, intercept_sigma, r_value_sigma, p_value_sigma, std_err_sigma = stats_utils.log_log_regression(sigma_function_all, run_sojourn_integral_all)

x_log10_range_sigma =  numpy.linspace(min(numpy.log10(sigma_function_all)) , max(numpy.log10(sigma_function_all)) , 10000)
#y_log10_fit_range = (slope*x_log10_range + intercept)

y_log10_sigma = (-0.5*x_log10_range_sigma + intercept_sigma)
#y_log10_low_tau_range = (0*x_log10_range + intercept)

#ax_sigma.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='--', zorder=2, label='Exponent = %.3f' % slope)

ax_sigma.plot(10**x_log10_range_sigma, 10**y_log10_sigma, c='k', lw=2.5, linestyle='--', zorder=2)#, label=r'$\tau \ll \mathcal{T}$')
#ax_sigma.plot(10**x_log10_range, 10**y_log10_low_tau_range, c='#87CEEB', lw=2.5, linestyle=':', zorder=2, label=r'$\tilde{\tau} \ll t, \mathcal{T}-t$')


ax_sigma.set_xlabel('Inverse environmental\nnoise, ' + r'$2\sigma^{-1} -2$', fontsize=11)
#bbox=dict(facecolor='white', edgecolor='none')
ax_sigma.set_ylabel(r'$\mathcal{A}(\mathcal{T})$', fontsize=12)

#inset_ax.set_xlabel('Inset X-label', bbox=dict(facecolor='white', edgecolor='none'))

#ax_sigma.legend(loc='lower left', fontsize=10)

#x_time.set_ylim([0.08, 50])

ax_sigma.set_xscale('log', base=10)
ax_sigma.set_yscale('log', base=10)

# for spine in ax_sigma.spines.values():
#    spine.set_edgecolor('black')   # Border color
#    spine.set_linewidth(1.5)       # Border thickness



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_vs_integral.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()



# 




