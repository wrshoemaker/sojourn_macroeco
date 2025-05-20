
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
target_dataset = 'david_et_al'
tau = 3


fig = plt.figure(figsize = (8.5, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

host_all = list(mle_dict[target_dataset].keys())
host_all.sort()


#ax_mean = plt.subplot2grid((1, 2), (0, 0))
#ax_cv = plt.subplot2grid((1, 2), (0, 1))

mean_all = []
cv_all = []
obs_mean_sojourn_all = []
pred_mean_sojourn_all = []
for host in host_all:

    for key, value in mle_dict[target_dataset][host].items():

        x_mean = value['x_mean']
        x_std = value['x_std']
        x_cv = x_std/x_mean

        max_sojourn_time = value['max_possible_sojourn_time']
        sojourn_time_range = numpy.arange(1, max_sojourn_time+1)

        k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
        #tau_ = 1+tau*sigma
        #tau_ = 1+tau*sigma

        sojourn_time_ou_pdf = theory_utils.predict_sojourn_dist_ou(max_sojourn_time, sigma, tau, data_utils.epsilon_fract_data, normalize=True)

        mean_all.append(x_mean)
        cv_all.append(x_cv)
        obs_mean_sojourn_all.append(numpy.mean(value['days_run_lengths']))
        pred_mean_sojourn_all.append(sum(sojourn_time_ou_pdf*sojourn_time_range))


def plot_cv_vs_mean_sojourn():

    fig, ax = plt.subplots(figsize=(6,4))

    #ax_mean.scatter(mean_all, obs_mean_sojourn_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.dataset_color_dict[target_dataset], label='ASV ' + r'$\times$' + ' host ' + r'$\times$' + ' sojourn')
    ax.scatter(cv_all, obs_mean_sojourn_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.dataset_color_dict[target_dataset], label='ASV ' + r'$\times$' + ' host ' + r'$\times$' + ' sojourn')

    slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = data_utils.stats.linregress(numpy.log10(mean_all), numpy.log10(obs_mean_sojourn_all))

    x_log10_range_mean =  numpy.linspace(min(numpy.log10(mean_all)) , max(numpy.log10(mean_all)) , 10000)
    y_log10_fit_mean = (slope_mean*x_log10_range_mean + intercept_mean)
    y_log10_pred_mean = x_log10_range_mean*0 + intercept_mean


    #ax_mean.plot(10**x_log10_range_mean, 10**y_log10_fit_mean, c='k', lw=2.5, linestyle=':', zorder=2, label="Regression")
    #ax_mean.plot(10**x_log10_range_mean, 10**y_log10_pred_mean, c='k', lw=3, linestyle='--', zorder=2, label="Prediction")

    # plot CV prediction
    cv_all_log10 = numpy.log10(cv_all)
    pred_mean_sojourn_all_log10 = numpy.log10(pred_mean_sojourn_all)
    hist_all, bin_edges_all = numpy.histogram(cv_all_log10, density=True, bins=10)
    bins_mean_all = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_mean_all_to_keep = []
    bins_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        y_i = pred_mean_sojourn_all_log10[(cv_all_log10>=bin_edges_all[i]) & (cv_all_log10<bin_edges_all[i+1])]
        
        if len(y_i) < 3:
            continue
        
        bins_mean_all_to_keep.append(bin_edges_all[i])
        bins_y.append(numpy.mean(y_i))


    bins_mean_all_to_keep = numpy.asarray(bins_mean_all_to_keep)
    bins_y = numpy.asarray(bins_y)

    bins_mean_all_to_keep_no_nan = bins_mean_all_to_keep[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_y))]
    bins_y_no_nan = bins_y[(~numpy.isnan(bins_mean_all_to_keep)) & (~numpy.isnan(bins_y))]


    ax.plot(10**bins_mean_all_to_keep_no_nan, 10**bins_y_no_nan, lw=3 , c='k', ls='--', label="Prediction")

    #ax_mean.set_xlabel("Mean relative abundance, " + r'$\bar{x}$', fontsize=14)
    ax.set_xlabel("CV of relative abundance, " + r'$\mathrm{CV}_{x}$', fontsize=14)

    #ax_mean.set_ylabel("Mean sojourn time (days), " + r'$\left < \mathcal{T} \right>$', fontsize=14)
    ax.set_ylabel("Mean sojourn time (days), " + r'$\left < \mathcal{T} \right>$', fontsize=14)


    y_lim_min = 1.3
    y_lim_max = 14

    ax.set_xlim([0.2, 2.01])

    #ax_mean.set_ylim([y_lim_min, y_lim_max])
    ax.set_ylim([y_lim_min, y_lim_max])

    #ax_mean.set_xscale('log', base=10)
    ax.set_xscale('log', base=10)

    #ax_mean.set_yscale('log', base=10)
    ax.set_yscale('log', base=10)


    ax.legend(loc = 'upper left', fontsize=11)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scv_vs_mean_sojourn_time.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





def plot_mean_vs_mean_sojourn():

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(mean_all, obs_mean_sojourn_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.dataset_color_dict[target_dataset], label='ASV ' + r'$\times$' + ' host ' + r'$\times$' + ' sojourn')

    slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = data_utils.stats.linregress(numpy.log10(mean_all), numpy.log10(obs_mean_sojourn_all))

    x_log10_range_mean =  numpy.linspace(min(numpy.log10(mean_all)) , max(numpy.log10(mean_all)) , 10000)
    y_log10_fit_mean = (slope_mean*x_log10_range_mean + intercept_mean)
    y_log10_pred_mean = x_log10_range_mean*0 + intercept_mean


    #ax.plot(10**x_log10_range_mean, 10**y_log10_fit_mean, c='k', lw=2.5, linestyle=':', zorder=2, label="Regression")
    ax.plot(10**x_log10_range_mean, 10**y_log10_pred_mean, c='k', lw=3, linestyle='--', zorder=2, label="Prediction")



    ax.set_xlabel("Mean relative abundance, " + r'$\bar{x}$', fontsize=14)
    ax.set_ylabel("Mean sojourn time (days), " + r'$\left < \mathcal{T} \right>$', fontsize=14)

    #y_lim_min = 1.3
    #y_lim_max = 14
    #ax.set_xlim([0.2, 2.01])
    #ax.set_ylim([y_lim_min, y_lim_max])

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.legend(loc = 'upper left', fontsize=11)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%smean_vs_mean_sojourn_time.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




#plot_cv_vs_mean_sojourn()
plot_mean_vs_mean_sojourn()


