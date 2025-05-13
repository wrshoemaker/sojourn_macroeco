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
from scipy.optimize import minimize_scalar


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
import theory_utils


environment = 'gut'
tau = 8


#max_sojourn_time = theory_utils.max_sojourn_time





fig = plt.figure(figsize = (12, 8)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

host_count = 0

for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

    ax_mean = plt.subplot2grid((2, len(data_utils.dataset_all)), (0, dataset_idx))
    ax_cv = plt.subplot2grid((2, len(data_utils.dataset_all)), (1, dataset_idx))
    #ax_cv_resid = plt.subplot2grid((2, len(data_utils.dataset_all)), (2, dataset_idx))

    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    mean_all_hosts = []
    cv_all_hosts = []
    obs_mean_sojourn_all_hosts = []
    pred_mean_sojourn_all_hosts = []
    for host in host_all:

        #print(mle_dict[dataset][host])

        mean_all_otu = []
        cv_all_otu = []
        obs_mean_sojourn_all_otu = []
        for key, value in mle_dict[dataset][host].items():

            x_mean = value['x_mean']
            x_std = value['x_std']
            x_cv = x_std/x_mean

            max_sojourn_time = value['max_possible_sojourn_time']
            sojourn_time_range = numpy.arange(1, max_sojourn_time+1)

            k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
            tau_ = 1+tau*sigma

            sojourn_time_ou_pdf = theory_utils.predict_sojourn_dist_ou(max_sojourn_time, sigma, tau_, data_utils.epsilon_fract_data, normalize=True)

            mean_all_otu.append(x_mean)
            cv_all_otu.append(x_cv)
            obs_mean_sojourn_all_otu.append(numpy.mean(value['days_run_lengths']))
            
            pred_mean_sojourn_all_hosts.append(sum(sojourn_time_ou_pdf*sojourn_time_range))


        mean_all_hosts.extend(mean_all_otu)
        cv_all_hosts.extend(cv_all_otu)
        obs_mean_sojourn_all_hosts.extend(obs_mean_sojourn_all_otu)
        
        ax_mean.scatter(mean_all_otu, obs_mean_sojourn_all_otu, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.host_color_dict[dataset][host])
        ax_cv.scatter(cv_all_otu, obs_mean_sojourn_all_otu, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.host_color_dict[dataset][host])


    #
    cv_all_hosts = numpy.asarray(cv_all_hosts)
    pred_mean_sojourn_all_hosts = numpy.asarray(pred_mean_sojourn_all_hosts)

    sort_cv_idx = numpy.argsort(cv_all_hosts)
    #print(sort_cv_idx)
    cv_all_hosts = cv_all_hosts[sort_cv_idx]
    pred_mean_sojourn_all_hosts = pred_mean_sojourn_all_hosts[sort_cv_idx]
    
    cv_all_hosts_log10 = numpy.log10(cv_all_hosts)
    pred_mean_sojourn_all_hosts_log10 = numpy.log10(pred_mean_sojourn_all_hosts)

    print(cv_all_hosts)

    # bin prediction
    hist_all, bin_edges_all = numpy.histogram(cv_all_hosts_log10, density=True, bins=10)
    bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_mean_all_to_keep = []
    bins_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        y_i = pred_mean_sojourn_all_hosts_log10[(cv_all_hosts_log10>=bin_edges_all[i]) & (cv_all_hosts_log10<bin_edges_all[i+1])]
        
        if len(y_i) < 5:
            continue
        
        bins_mean_all_to_keep.append(bin_edges_all[i])
        bins_y.append(numpy.mean(y_i))


    bins_mean_all_to_keep = numpy.asarray(bins_mean_all_to_keep)
    bins_y = numpy.asarray(bins_y)

    bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_y))]
    bins_y_no_nan = bins_y[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_y))]

    ax_mean.set_xlabel("Mean relative abundance, " + r'$\bar{x}$', fontsize=12)
    ax_cv.set_xlabel("CV of relative abundance, " + r'$\mathrm{CV}_{x}$', fontsize=12)

    ax_mean.set_ylabel("Mean sojourn time, " + r'$\left < \mathcal{T} \right>$', fontsize=12)
    ax_cv.set_ylabel("Mean sojourn time, " + r'$\left < \mathcal{T} \right>$', fontsize=12)


    ax_cv.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_y_no_nan, lw=2, c='k', ls='--')


    ax_mean.set_xscale('log', base=10)
    ax_cv.set_xscale('log', base=10)

    ax_mean.set_yscale('log', base=10)
    ax_cv.set_yscale('log', base=10)


    slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = data_utils.stats.linregress(numpy.log10(mean_all_hosts), numpy.log10(obs_mean_sojourn_all_hosts))
    slope_cv, intercept_cv, r_value_cv, p_value_cv, std_err_cv = data_utils.stats.linregress(numpy.log10(cv_all_hosts), numpy.log10(obs_mean_sojourn_all_hosts))

    


    print(slope_mean, p_value_mean, slope_cv, p_value_cv)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%scv_vs_mean_sojourn_time.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()


