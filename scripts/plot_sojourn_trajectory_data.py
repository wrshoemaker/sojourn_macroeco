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
from matplotlib import cm, colors, colorbar

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


import stats_utils
import simulation_utils



mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

min_n_sojourn_intermediate_obs = 3


def plot_sojourn_trajectory_data():

    n_rows = len(data_utils.dataset_all)
    n_cols = 4
    fig = plt.figure(figsize = (16, 12)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    cmap = cm.ScalarMappable(norm = colors.Normalize(1, 40), cmap = plt.get_cmap('Blues'))

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host_idx, host in enumerate(host_all):
        
            sys.stderr.write("Analyzing host %s.....\n" % host)

            run_length_all = []
            run_sojourn_integral_all = []

            ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
            ax.set_title('%s, %s' % (dataset, host), fontsize=12)

            run_sojourn_all = []

            for asv in  mle_dict[dataset][host].keys():
                
                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict is None:
                    continue
                
                for run_length, run_sojourn in run_dict.items():
                    
                    for run_sojourn_j in run_sojourn:

                        ax.plot(numpy.linspace(0, 1, num=len(run_sojourn_j), endpoint=True), run_sojourn_j, lw=1, alpha=1, c=cmap.to_rgba(run_length), ls='-')

                        run_sojourn_all.extend(run_sojourn_j)


            ax.set_xlim([0,1])
            ax.set_ylim([0,max(run_sojourn_all)*1.1])

            if (host_idx == 0):
                ax.set_ylabel("Sojourn deviation, " + r'$ y(t) - \left < y \right >$', fontsize=12)
                
            # x-label
            if (dataset_idx == len(data_utils.dataset_all)-1):
                ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=12)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_trajectory_data.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_mean_sojourn_trajector_data():

    mean_run_dict = {}
    sigma_all = []

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        #sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host_idx, host in enumerate(host_all):
        
            #sys.stderr.write("Analyzing host %s.....\n" % host)
            run_length_all = []
            run_sojourn_integral_all = []

            run_sojourn_all = []

            for asv in  mle_dict[dataset][host].keys():
                
                run_dict = mle_dict[dataset][host][asv]['run_dict']
                x_mean = mle_dict[dataset][host][asv]['x_mean']
                x_std = mle_dict[dataset][host][asv]['x_std']
                x_cv = x_mean/x_std

                k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)

                if run_dict is None:
                    continue
                
                for run_length, run_sojourn in run_dict.items():
                    
                    for run_sojourn_j in run_sojourn:

                        if run_length not in mean_run_dict:
                            mean_run_dict[run_length] = {}
                            mean_run_dict[run_length]['run_sojourn_all'] = []
                            mean_run_dict[run_length]['asv_all'] = []
                            mean_run_dict[run_length]['sigma_all'] = []

                        mean_run_dict[run_length]['run_sojourn_all'].append(run_sojourn_j)
                        mean_run_dict[run_length]['asv_all'].append(asv)
                        mean_run_dict[run_length]['sigma_all'].append(sigma)

    
    run_length_all = list(mean_run_dict.keys())
    run_length_all.sort()


    run_length_final = []
    run_sojourn_merged_final = []
    sigma_final = []
    for run_length in run_length_all:

        if len(mean_run_dict[run_length]['run_sojourn_all']) < 2:
            continue
        
        run_sojourn_all = mean_run_dict[run_length]['run_sojourn_all']
        sigma_final.extend(mean_run_dict[run_length]['sigma_all'])
        # get the relative time within sojourn period for all sojourns
        s_range_all = [numpy.linspace(0, 1, num=len(n),  endpoint=True) for n in run_sojourn_all]
        
        s_range_all_flat = numpy.concatenate(s_range_all).ravel()
        run_sojourn_all_flat = numpy.concatenate(run_sojourn_all).ravel()

        #print(run_sojourn_all_flat)

        len_run_sojourn_all = [len(s) for s in run_sojourn_all]
        len_run_sojourn_all.sort()
        min_len_run_sojourn = min(len_run_sojourn_all)
        s_range_to_plot = numpy.linspace(0, 1, num=len_run_sojourn_all[1], endpoint=True)[1:]

        # start and end 
        run_sojourn_start = numpy.mean(run_sojourn_all_flat[s_range_all_flat==0])
        run_sojourn_end = numpy.mean(run_sojourn_all_flat[s_range_all_flat==1])

        run_sojourn_intermediate = []
        for s_idx in range(len(s_range_to_plot)-1):

            run_sojourn_intermediate_s = run_sojourn_all_flat[(s_range_all_flat > s_range_to_plot[s_idx]) & (s_range_all_flat < s_range_to_plot[s_idx+1])]
            
            if len(run_sojourn_intermediate_s) >= min_n_sojourn_intermediate_obs:
                run_sojourn_intermediate.append(numpy.mean(run_sojourn_intermediate_s))

        if len(run_sojourn_intermediate) < 3:
            continue
        
        run_sojourn_merged = [run_sojourn_start] + run_sojourn_intermediate + [run_sojourn_end]

        run_length_final.append(run_length)
        run_sojourn_merged_final.append(run_sojourn_merged)


    
    #asv_final = list(set(asv_final))

    #x_mean_final = mle_dict[dataset][host][asv]['x_mean']
    #x_std = mle_dict[dataset][host][asv]['x_std']
    #x_cv = x_mean/x_std
    #print(asv_final)

    sigma_final = numpy.asarray(list(set(sigma_final)))
    mean_sigma = numpy.mean(sigma_final)
    predicted_sojourn = (((2/sigma_final) - 1)**(-0.5)) * numpy.sqrt(8/numpy.pi) 


    #print(numpy.mean(predicted_sojourn))

    cmap = cm.ScalarMappable(norm = colors.Normalize(1, max(run_length_final)+1), cmap = plt.get_cmap('Blues'))
    fig, ax = plt.subplots(figsize=(5,4))

    run_sojourn_merged_final_flat = []
    for r_idx, r in enumerate(run_length_final):
        run_sojourn_merged_final_flat.extend(run_sojourn_merged_final[r_idx])
        ax.plot(numpy.linspace(0, 1, num=len(run_sojourn_merged_final[r_idx]), endpoint=True), run_sojourn_merged_final[r_idx], lw=1, alpha=1, c=cmap.to_rgba(r), ls='-')

    ax.set_xlim([0,1])
    ax.set_ylim([0,max(run_sojourn_merged_final_flat)*1.1])
    ax.set_xlabel('Rescaled time within sojourn period, ' + r'$\frac{t}{\mathcal{T}}$', fontsize=15)
    ax.set_ylabel("Mean sojourn deviation, " + r'$ \left < y(t) - \bar{y} \right >_{\mathcal{T}}$', fontsize=15)

    # colorbar
    #cmap.set_array([])  # Required for colorbar in some versions
    cmap.set_array(numpy.linspace(1,  max(run_length_final)+1, 100))

    # Plot colorbar on top
    cbar_ax = fig.add_axes([0.15, 0.92, 0.7, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(cmap, cax=cbar_ax, orientation='horizontal', pad=0.1)

    # Move it to the top
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    
    cbar.set_label("Sojourn time (days), "  + r'$\mathcal{T}$', fontsize=15)

    cbar.set_ticks([1, 5, 10, 15, 20])       # Set tick positions
    cbar.set_ticklabels(['1', '5', '10', '15', '20'])
    cbar.ax.tick_params(labelsize=8)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%smean_sojourn_trajector_data.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



plot_mean_sojourn_trajector_data()