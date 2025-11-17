
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
from matplotlib.colors import LinearSegmentedColormap

import stats_utils
import simulation_utils

#lol
#import wesanderson
#cmap_species = wesanderson.film_palette("The Life Aquatic with Steve Zissou")


trajectory_dict_path = '%strajectory_dict.pickle' %config.data_directory


numpy.random.seed(123456789)

t_total = 200
t_range = numpy.arange(t_total)
n_asvs = 20
sigma=1
tau = 4
target_asv_color = '#eb5900'
#'darkred':              '#8B0000',



lw_target=1.8
alpha_target=1

label_fontsize=15


def make_example_asv_trajectory_dict():

    log10_k_all = numpy.random.uniform(1, 7, n_asvs)
    asv_trajectory_all = []
    for log10_k in log10_k_all:

        asv_trajectory = simulation_utils.simulate_slm_trajectory(t_total=t_total, n_reps=1, k=10**log10_k, sigma=sigma, tau=tau, epsilon=0.001)[1:,0]
        #def simulate_slm_trajectory(t_total=None, n_reps=100, k=10000, sigma=1, tau=7, epsilon=0.001, analytic=False, init_log=False, return_log=False):
        asv_trajectory_all.append(asv_trajectory)

    # calculate relative abundances
    # save nested list as pickle
    trajectory_dict = {}
    trajectory_dict['log10_k_all'] = log10_k_all.tolist()
    trajectory_dict['asv_trajectory_all'] = asv_trajectory_all
    # save pickle

    sys.stderr.write("Saving dictionary...\n")
    with open(trajectory_dict_path, 'wb') as outfile:
        pickle.dump(trajectory_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




def plot_example_fig1():

    #make_example_asv_trajectory_dict()
    # load the dictionary
    trajectory_dict = pickle.load(open(trajectory_dict_path, "rb"))


    asv_trajectory_all = numpy.stack(trajectory_dict['asv_trajectory_all'])
    rel_asv_trajectory_all = asv_trajectory_all/numpy.sum(asv_trajectory_all, axis=0)

    color_palette = ['#3B9AB2', '#78B7C5', '#EBCC2A', '#E1AF00', '#F21A00']
    cmap = LinearSegmentedColormap.from_list('life_aquatic', color_palette, N=n_asvs)
    #target_asv_color = '#e49700'

    #target_asv_idx = 15
    target_asv_idx = 17
    target_asv_trajectory = rel_asv_trajectory_all[target_asv_idx]
    target_asv_k = 10**trajectory_dict['log10_k_all'][target_asv_idx]


    fig, ax = plt.subplots(figsize=(7.5,3))
    for rel_asv_trajectory_idx, rel_asv_trajectory in enumerate(rel_asv_trajectory_all):
        cmap_i = cmap(rel_asv_trajectory_idx)
        cmap_hex_i = colors.to_hex(cmap_i)    

        if rel_asv_trajectory_idx == target_asv_idx:
            lw=lw_target
            alpha=alpha_target

        else:
            lw=0.6
            alpha=0.5

        ax.plot(t_range, rel_asv_trajectory, alpha=alpha, lw=lw, c=cmap_i)


    ax.set_xlim([1,t_total])
    ax.set_ylim([4*(10**-8),1])
    ax.set_yscale('log', base=10)

    ax.set_xlabel("Time (days), " + r'$t$', fontsize=label_fontsize)
    ax.set_ylabel("Relative abundance, " + r'$x(t)$', fontsize=label_fontsize)

    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sex_asv_trajectory.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()


    #fig, ax = plt.subplots(figsize=(7.5,3))
    fig, ax = plt.subplots(figsize=(16,3))

    # get parameters of target trajectory
    mean_x = numpy.mean(target_asv_trajectory)
    std_x = numpy.std(target_asv_trajectory)
    cv_x = std_x/mean_x

    # rescale
    #mean_gamma, cv_gamma = simulation_utils.calculate_mean_and_cv_slm(target_asv_k, sigma)

    log_rescaled_target_asv_trajectory = numpy.log(target_asv_trajectory/mean_x)
    #print(numpy.mean(log_rescaled_target_asv_trajectory))
    #mean_log_gamma = stats_utils.expected_value_log_gamma(mean_x, cv_x)
    diff_log_rescaled_target_asv_trajectory = log_rescaled_target_asv_trajectory - numpy.mean(log_rescaled_target_asv_trajectory)


    ax.plot(t_range, diff_log_rescaled_target_asv_trajectory, alpha=alpha_target, lw=lw_target, c=target_asv_color)
    ax.set_xlim([1,t_total])
    ax.set_ylim([-3.2,3.2])
    ax.axhline(y=0, lw=2, ls='--', c='k')



    t_area_start = 126
    t_area_stop = 159
    t_range_area = numpy.arange(t_area_start, t_area_stop)
    ax.fill_between(t_range_area, 0, diff_log_rescaled_target_asv_trajectory[t_area_start:t_area_stop], color=target_asv_color, alpha=0.4, zorder=1)


    ax.set_title('Temporal dynamics of a single community member', fontsize=16)
    ax.set_xlabel("Time (days), " + r'$t$', fontsize=label_fontsize)
    ax.set_ylabel("Deviation from typical\n" + r'$\mathrm{log}_{e}$' + " abundance, " + r'$y(t) - \bar{y}$', fontsize=label_fontsize)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.yaxis.set_tick_params(labelsize=7)




    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sex_asv_trajectorry_rescaled.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()


    # plot notation

    fig, ax = plt.subplots(figsize=(7.5,3))


    ax.text(0.05, 0.7, "Sojourn trajectory area, " +  r'$\mathcal{A}(\mathcal{T}\, ) \equiv \int_{0}^{1} \left [ y(s \cdot \mathcal{T} \, ) - \bar{y} \right ] \, ds$', fontsize=15, transform=ax.transAxes)
    ax.text(0.05, 0.5, "Sojourn time, " +  r'$\mathcal{T}$', fontsize=15, transform=ax.transAxes)
    ax.text(0.01, 0.3, "Typical deviation within sojourn trajectory, " +  r'$\left < y(t) - \bar{y} \right >_{\mathcal{T}}$', fontsize=15, transform=ax.transAxes)

    ax.text(0.01, 0.1, r'$\sigma$', fontsize=15, color=target_asv_color, transform=ax.transAxes)
    ax.text(0.3, 0.1, r'$>$', fontsize=15, color='k', transform=ax.transAxes)
    ax.text(0.5, 0.1, r'$\sigma$', fontsize=15, color='darkred', transform=ax.transAxes)

    ax.text(0.01, 0.2, r'$K$', fontsize=15, color=target_asv_color, transform=ax.transAxes)
    ax.text(0.3, 0.2, r'$=$', fontsize=15, color='k', transform=ax.transAxes)
    ax.text(0.5, 0.2, r'$K$', fontsize=15, color='darkred', transform=ax.transAxes)


    ax.text(0.35, 0.9, "a", fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.9, "b", fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.text(0.6, 0.9, "c", fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.text(0.8, 0.9, "d", fontsize=15, fontweight='bold', transform=ax.transAxes)




    #fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%snotation.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_example_fig2():

    log10_k = 5
    sigma_1 = 0.6
    sigma_2 = 1

    asv_trajectory_1 = simulation_utils.simulate_slm_trajectory(t_total=t_total, n_reps=1, k=10**log10_k, sigma=sigma_1, tau=tau, epsilon=0.001)[1:,0]
    asv_trajectory_2 = simulation_utils.simulate_slm_trajectory(t_total=t_total, n_reps=1, k=10**log10_k, sigma=sigma_2, tau=tau, epsilon=0.001)[1:,0]


    log_rescaled_asv_trajectory_1 = numpy.log(asv_trajectory_1/numpy.mean(asv_trajectory_1))
    diff_log_rescaled_asv_trajectory_1 = log_rescaled_asv_trajectory_1 - numpy.mean(log_rescaled_asv_trajectory_1)

    log_rescaled_asv_trajectory_2 = numpy.log(asv_trajectory_2/numpy.mean(asv_trajectory_2))
    diff_log_rescaled_asv_trajectory_2 = log_rescaled_asv_trajectory_2 - numpy.mean(log_rescaled_asv_trajectory_2)


    fig, ax = plt.subplots(figsize=(6,3))

    ax.plot(t_range, diff_log_rescaled_asv_trajectory_1, alpha=1, lw=2, c=target_asv_color)
    ax.plot(t_range, diff_log_rescaled_asv_trajectory_2, alpha=1, lw=2, c='darkred')

    ax.set_xlim([1,89])
    ax.set_ylim([-3.2,3.2])
    ax.axhline(y=0, lw=2, ls='--', c='k')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
   

    ax.set_xlabel("Time (days), " + r'$t$', fontsize=label_fontsize)
    ax.set_ylabel("Deviation from typical\n" + r'$\mathrm{log}_{e}$' + " abundance, " + r'$y(t) - \bar{y}$', fontsize=label_fontsize)




    fig_name = "%sfig2_conceptual.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




#make_example_asv_trajectory_dict()
plot_example_fig1()
#plot_example_fig2()
