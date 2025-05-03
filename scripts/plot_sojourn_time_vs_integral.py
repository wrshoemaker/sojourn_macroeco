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

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))



fig, ax = plt.subplots(figsize=(6,4))

run_length_all_all = []
run_sojourn_integral_all_all = []
for dataset in data_utils.dataset_all:

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host in host_all:

        sys.stderr.write("Analyzing host %s.....\n" % host)

        run_length_all = []
        run_sojourn_integral_all = []

        for asv in  mle_dict[dataset][host].keys():
            
            run_dict = mle_dict[dataset][host][asv]['run_dict']

            if run_dict is None:
                continue
            
            for run_length, run_sojourn in run_dict.items():

                for run_sojourn_j in run_sojourn:

                    run_sojourn_j = numpy.asarray(run_sojourn_j)

                    print(len(run_sojourn_j))

                    run_sojourn_integral_j = integrate.simpson(run_sojourn_j, numpy.linspace(0, 1, num=len(run_sojourn_j), endpoint=True))

                    # run length has units of days
                    run_length_all.append(run_length)
                    run_sojourn_integral_all.append(run_sojourn_integral_j)

                    run_length_all_all.append(run_length)
                    run_sojourn_integral_all_all.append(run_sojourn_integral_j)


        if len(run_length_all) == 0:
            continue
        
        ax.scatter(run_length_all, run_sojourn_integral_all, color=plot_utils.host_color_dict[dataset][host], alpha=0.6, s=10)



ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

# regression...

slope, intercept, r_value, p_value, std_err = stats_utils.log_log_regression(run_length_all_all, run_sojourn_integral_all_all)

x_log10_range =  numpy.linspace(min(numpy.log10(run_length_all_all)) , max(numpy.log10(run_length_all_all)) , 10000)
y_log10_fit_range = (slope*x_log10_range + intercept)

ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2)

ax.text(0.5, 0.1, 'Exponent = %.3f' % slope, fontsize=13, transform=ax.transAxes)


ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=12)
ax.set_ylabel("Cumulative walk length, " + r'$\int_{0}^{1}  \tilde{y}_{s\mathcal{T}} $', fontsize=12)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_vs_integral.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()