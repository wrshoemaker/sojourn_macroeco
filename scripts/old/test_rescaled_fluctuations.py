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

numpy.random.seed(123456789)

import pickle


environment = 'gut'
occupancy_min = 1.0


ax_idx_all = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1), (4,0), (4,1)]

slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sim_dict_path = '%sfluctuation_sim_dict.pickle' %config.data_directory







def plot_sojourn_time_vs_cumulative_walk_length(log=True):

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, environment)

        host_all = list(set(host_status))
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            # function subsets ASVs that are actually present
            read_counts_host, days_host, asv_names_host = data_utils.subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)
            rel_read_counts_host = (read_counts_host/read_counts_host.sum(axis=0))

            occupancy_min_idx = numpy.sum(rel_read_counts_host>0, axis=1)/len(days_host) >= occupancy_min   

            rel_read_counts_host_subset = rel_read_counts_host[occupancy_min_idx,:]
            asv_names_host_subset = asv_names_host[occupancy_min_idx]

            if log == True:
                rel_read_counts_host_subset_log = numpy.log(rel_read_counts_host_subset)
                diff_rel_read_counts_host_subset = (rel_read_counts_host_subset_log.T - rel_read_counts_host_subset_log[:,0]).T

            else:
                diff_rel_read_counts_host_subset = (rel_read_counts_host_subset.T - rel_read_counts_host_subset[:,0]).T

            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])

            run_length_all = []
            run_sum_all = []
            for afd_i_idx in range(len(asv_names_host_subset)):
                
                diff_trajectory_i = diff_rel_read_counts_host_subset[afd_i_idx,:]
    
                run_values, run_starts, run_lengths = find_runs(diff_trajectory_i>0)

                # Aaverage trajectory of walker before it returns to starting position x(0)
                # is obtained by summing over all positive walks starting at time t=0,
                # constrained to return for the **first time** to x(0) at time t = T

                # ASV never returns to the origin
                if len(run_values) == 1:
                    continue

                for run_j_idx in range(len(run_values)):
                    
                    # minimum of 10 samples
                    days_run_j = days_host[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]]

                    # only one observation.
                    if len(days_run_j) == 1:
                        continue

                    days_run_length_j = days_run_j[-1] - days_run_j[0]
                    run_j = numpy.absolute(diff_trajectory_i[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]])

                    run_length_all.append(days_run_length_j)
                    run_sum_all.append(sum(run_j))


            run_length_all_array = numpy.asarray(run_length_all)
            run_sum_all_array = numpy.asarray(run_sum_all)

            run_length_min = 10
            run_length_all_array_filtered = run_length_all_array[run_length_all_array >= run_length_min]
            run_sum_all_array_filtered = run_sum_all_array[run_length_all_array >= run_length_min]

            slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_all_array_filtered), numpy.log10(run_sum_all_array_filtered))

            x_log10_range =  numpy.linspace(min(numpy.log10(run_length_all_array)) , max(numpy.log10(run_length_all_array)) , 10000)
            y_log10_fit_range = (slope*x_log10_range + intercept)

            ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
            ax.scatter(run_length_all, run_sum_all, s=10, alpha=0.4, c='k', zorder=1)

            ax.text(0.5, 0.1, 'Slope = %.2f' % slope, fontsize=13, transform=ax.transAxes)

            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)

            ax.set_xlabel('Sojourn time (days), ' + r'$T$', fontsize=10)
            ax.set_ylabel('Cumulative walk length, ' + r'$\sum_{t \in T} x(t)$', fontsize=10)

            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            
            host_count+=1


    #if log==True:
    #    log_str = '_log'
    #else:
    #   log_str = ''

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_vs_cumulative_walk_length.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_rescaled_fluctuations(log=True):

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, environment)

        host_all = list(set(host_status))
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            # function subsets ASVs that are actually present
            read_counts_host, days_host, asv_names_host = data_utils.subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)
            rel_read_counts_host = (read_counts_host/read_counts_host.sum(axis=0))

            occupancy_min_idx = numpy.sum(rel_read_counts_host>0, axis=1)/len(days_host) >= occupancy_min   

            rel_read_counts_host_subset = rel_read_counts_host[occupancy_min_idx,:]
            asv_names_host_subset = asv_names_host[occupancy_min_idx]

            # dynamics are stationary, so we rescale using the mean 
            
            if log == True:
                rel_read_counts_host_subset_log = numpy.log(rel_read_counts_host_subset)
                diff_rel_read_counts_host_subset = (rel_read_counts_host_subset_log.T - rel_read_counts_host_subset_log[:,0]).T

            else:
                diff_rel_read_counts_host_subset = (rel_read_counts_host_subset.T - rel_read_counts_host_subset[:,0]).T


            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            rescaled_days_run_all = []
            rescaled_run_all = []
            for afd_i_idx in range(len(asv_names_host_subset)):
                
                diff_trajectory_i = diff_rel_read_counts_host_subset[afd_i_idx,:]
    
                run_values, run_starts, run_lengths = find_runs(diff_trajectory_i>0)

                print(dataset, host, numpy.mean(run_lengths), len(run_lengths))

                # Aaverage trajectory of walker before it returns to starting position x(0)
                # is obtained by summing over all positive walks starting at time t=0,
                # constrained to return for the **first time** to x(0) at time t = T

                # ASV never returns to the origin
                if len(run_values) == 1:
                    continue

                for run_j_idx in range(len(run_values)):
                    
                    # minimum of 10 samples
                    run_length_j = run_lengths[run_j_idx]
                    days_run_j = days_host[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]]

                    # only one observation.
                    if len(days_run_j) == 1:
                        continue
  
                    # ensure some minimum number of samples
                    if run_length_j < 40:
                        continue

                    days_run_length_j = days_run_j[-1] - days_run_j[0]

                    #print(afd_i_idx, days_run_length_j)

                    # absolute value due to symmetry..
                    run_j = numpy.absolute(diff_trajectory_i[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]])
                    rescaled_days_run_j = (days_run_j - days_run_j[0])/days_run_length_j

                    rescaled_run_j = run_j/days_run_length_j
                    #ax.scatter(rescaled_days_run_j, run_j, s=15, alpha=0.6, c='k')
                    ax.plot(rescaled_days_run_j, rescaled_run_j, lw=0.5, alpha=0.4, c='k', ls='-')

                    rescaled_days_run_all.append(rescaled_days_run_j)
                    rescaled_run_all.append(rescaled_run_j)

 

            rescaled_days_run_all = numpy.concatenate(rescaled_days_run_all).ravel()
            rescaled_run_all = numpy.concatenate(rescaled_run_all).ravel()

            bins_x_to_keep, bins_y = plot_utils.get_bin_x_mean_y(rescaled_days_run_all, rescaled_run_all, bins=14, min_n_in_bin=5)
            ax.plot(bins_x_to_keep, bins_y, lw=4, alpha=1, c='r', ls=':', zorder=3)

            ax.set_xlabel('Time rescaled by total sojourn time, ' + r'$t/T$', fontsize=12)
            ax.set_ylabel(r'$(x(t) - x(0)) \cdot T^{-1}$', fontsize=12)

            if host_count == 0:

                legend_elements = [Line2D([0], [0], color='r', lw=4, label='Mean over ASVs and sojourns'),
                                    Line2D([0], [0], color='k', lw=4, label='(ASV ' + r'$\times$' +  'sojourn)')]

                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

            #ax.set_yscale('log')
            host_count+=1

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_fluctuations.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()
                


def plot_dist_run_lengths():

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, environment)

        host_all = list(set(host_status))
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            # function subsets ASVs that are actually present
            read_counts_host, days_host, asv_names_host = data_utils.subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)
            rel_read_counts_host = (read_counts_host/read_counts_host.sum(axis=0))

            occupancy_min_idx = numpy.sum(rel_read_counts_host>0, axis=1)/len(days_host) >= occupancy_min   

            rel_read_counts_host_subset = rel_read_counts_host[occupancy_min_idx,:]
            asv_names_host_subset = asv_names_host[occupancy_min_idx]

            # dynamics are stationary, so we rescale using the mean 
            
  
            diff_rel_read_counts_host_subset = (rel_read_counts_host_subset.T - rel_read_counts_host_subset[:,0]).T


            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            #rescaled_days_run_all = []
            days_run_lengths_all = []
            run_values_all = []
            for afd_i_idx in range(len(asv_names_host_subset)):
                
                diff_trajectory_i = diff_rel_read_counts_host_subset[afd_i_idx,:]
    
                run_values, run_starts, run_lengths = find_runs(diff_trajectory_i>0)

                # ASV never returns to the origin
                if len(run_values) == 1:
                    continue

                # calculate run lengths in terms of days
                run_ends = run_starts+run_lengths
                
                # never returned to origin, remove this run
                if run_ends[-1] == len(days_host):

                    run_starts = run_starts[:-1]
                    run_ends = run_ends[:-1]
                    run_values = run_values[:-1]


                days_run_length_i = days_host[run_ends] - days_host[run_starts]
                days_run_lengths_all.extend(days_run_length_i.tolist())
                run_values_all.extend(run_values.tolist())


            days_run_lengths_all = numpy.asarray(days_run_lengths_all)
            run_values_all = numpy.asarray(run_values_all)

            days_run_lengths_all_pos = days_run_lengths_all[run_values_all==True]
            days_run_lengths_all_neg = days_run_lengths_all[run_values_all==False]

            #days_run_lengths_rescaled_all = days_run_lengths_all / (max(days_host) - min(days_host))
            ax.hist(days_run_lengths_all_pos, bins=10, density=True, histtype='step', fill=False, color='b')
            ax.hist(days_run_lengths_all_neg, bins=10, density=True, histtype='step', fill=False, color='r')

            ax.set_xlabel("Sojourn time (days)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)

            ax.set_yscale('log', base=10)
            ax.set_xlim([1, 350])

            #if host_count == 0:

            #    legend_elements = [Line2D([0], [0], color='r', lw=4, label='Mean over ASVs and sojourns'),
            #                        Line2D([0], [0], color='k', lw=4, label='(ASV ' + r'$\times$' +  'sojourn)')]

                #ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

            #ax.set_yscale('log')
            host_count+=1


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdist_run_lengths.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()







    



def plot_sojourn_time_vs_cumulative_walk_length_demog(m, r, D, n_samples):

    abundances = simulate_demog_trajectory(m, r, D, n_samples, delta_t = 7)

    abundances_burn_in = abundances[100:]
    
    diff = abundances_burn_in - abundances_burn_in[0]

    run_values, run_starts, run_lengths = find_runs(diff>0)

    # cumulative walks
    run_sum_all = []
    for run_j_idx in range(len(run_values)):

        run_j = numpy.absolute(diff[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]])

        run_sum_all.append(sum(run_j))


    run_sum_all = numpy.asarray(run_sum_all)

    run_length_min = 10

    run_length_all_filter = run_lengths[run_lengths >= run_length_min]
    run_sum_all_filter = run_sum_all[run_lengths >= run_length_min]

    slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_all_filter), numpy.log10(run_sum_all_filter))

    fig, ax = plt.subplots(figsize=(6,4))

    x_log10_range =  numpy.linspace(min(numpy.log10(run_lengths)) , max(numpy.log10(run_lengths)) , 10000)
    y_log10_fit_range = (slope*x_log10_range + intercept)

    ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
    ax.scatter(run_lengths, run_sum_all, s=10, alpha=0.4, c='k', zorder=1)

    ax.text(0.5, 0.1, 'Slope = %.2f' % slope, fontsize=13, transform=ax.transAxes)

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel('Sojourn time (days), ' + r'$T$', fontsize=10)
    ax.set_ylabel('Cumulative walk length', fontsize=10)

    ax.set_title("Mig. + linear birth + demog. noise SDE (Dornic et al. method)", fontsize=11)

    
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_vs_cumulative_walk_length_demog.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_sojourn_time_vs_cumulative_walk_length_slm(K = 10000, sigma = 1, tau = 7, n_samples=100000):

    q_trajaectory = simulate_slm_trajectory(n_samples, K, sigma, tau, delta_t=1)
    x_trajectory = numpy.exp(q_trajaectory)

    diff = x_trajectory - x_trajectory[0]
    run_values, run_starts, run_lengths = find_runs(diff>0)

    run_sum_all = numpy.asarray([sum(numpy.absolute(diff[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]])) for run_j_idx in range(len(run_values))])
    run_length_min = 10

    run_length_all_filter = run_lengths[run_lengths >= run_length_min]
    run_sum_all_filter = run_sum_all[run_lengths >= run_length_min]

    slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_all_filter), numpy.log10(run_sum_all_filter))

    fig, ax = plt.subplots(figsize=(6,4))

    x_log10_range =  numpy.linspace(min(numpy.log10(run_lengths)) , max(numpy.log10(run_lengths)) , 10000)
    y_log10_fit_range = (slope*x_log10_range + intercept)

    ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
    ax.scatter(run_lengths, run_sum_all, s=10, alpha=0.4, c='k', zorder=1)

    ax.text(0.5, 0.1, 'Slope = %.2f' % slope, fontsize=13, transform=ax.transAxes)

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel('Sojourn time (days), ' + r'$T$', fontsize=10)
    ax.set_ylabel('Cumulative walk length', fontsize=10)

    ax.set_title("SLM SDE (Euler–Maruyama method)", fontsize=11)

    
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_vs_cumulative_walk_length_slm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()







#def plot_slope_sim():





def simulate_universality(n_samples, n_iter=100, min_run_length=100):

    sigma_range = numpy.logspace(-3, numpy.log10(2), num=10, endpoint=False, base=10)
    #D_range = numpy.logspace(-3, 1, num=10, endpoint=True, base=10)

    sigma_range = [0.01, 0.1, 1]
    D_range = [0.01, 0.1, 1]

    k = 1000
    tau = 20

    m = 100
    r = 0.1

    result_dict = {}
    result_dict['slm'] = {}
    result_dict['demog'] = {}

    for sigma in sigma_range:

        print("Sigma ", sigma)

        result_dict['slm'][sigma] = {}
        result_dict['slm'][sigma]['sojourn_time'] = []
        result_dict['slm'][sigma]['rescaled_diff_trajectories'] = []
        result_dict['slm'][sigma]['run_values'] = []
        result_dict['slm'][sigma]['slope'] = []

        while len(result_dict['slm'][sigma]['slope']) < n_iter:

            #while len(result_dict['slm'][sigma][k][tau]['slope']) < n_iter:
            q_trajectory_slm = simulate_slm_trajectory(n_samples, k, sigma, tau, delta_t=1)
            x_trajectory_slm = numpy.exp(q_trajectory_slm)
            diff_slm = (x_trajectory_slm - x_trajectory_slm[0])[1:]
            run_values_slm, run_starts_slm, run_lengths_slm = find_runs(diff_slm>0)
            
            run_sum_slm = numpy.asarray([sum(numpy.absolute(diff_slm[run_starts_slm[run_j_idx]:run_starts_slm[run_j_idx]+run_lengths_slm[run_j_idx]])) for run_j_idx in range(len(run_lengths_slm))])
            
            to_keep_idx = (run_lengths_slm >= min_run_length) & (run_sum_slm>0)
            
            run_values_slm_filter = run_values_slm[to_keep_idx]
            run_starts_slm_filter = run_starts_slm[to_keep_idx]
            run_length_slm_filter = run_lengths_slm[to_keep_idx]
            run_sum_slm_filter = run_sum_slm[to_keep_idx]

            # only fit regressions if there are at least 10 samples.
            if len(run_sum_slm_filter) < 10:
                continue

            slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_slm_filter), numpy.log10(run_sum_slm_filter))
        
            # we think > 2 are fitting issues
            if (slope < 0) or (slope > 2):
                continue

            for trajectory_i_idx in range(len(run_values_slm_filter)):

                run_starts_slm_filter_i = run_starts_slm_filter[trajectory_i_idx]
                run_length_slm_filter_i = run_length_slm_filter[trajectory_i_idx]

                trajectory_i = numpy.absolute(diff_slm[run_starts_slm_filter_i:(run_starts_slm_filter_i + run_length_slm_filter_i)])

                rescaled_trajectory_i = trajectory_i/(run_length_slm_filter_i**slope)
                result_dict['slm'][sigma]['rescaled_diff_trajectories'].append(rescaled_trajectory_i.tolist())
            

            result_dict['slm'][sigma]['run_values'].extend(run_values_slm.tolist())
            result_dict['slm'][sigma]['sojourn_time'].extend(run_lengths_slm.tolist())
            result_dict['slm'][sigma]['slope'].append(slope)

    
    #fig, ax = plt.subplots(figsize=(6,4))

    #x_trajectory_demog = simulate_demog_trajectory(m, r, D, n_samples, delta_t=1)
        
    #ax.plot(range(len(x_trajectory_demog))[:1000], x_trajectory_demog[:1000], lw=1,alpha=0.6)
    #ax.hist(run_lengths_slm, density=True, bins=20)
    #ax.set_yscale('log', base=10)

    #fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #fig_name = "%ssimulate_sojourn_dist.png" % (config.analysis_directory)
    #fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    #plt.close()

    for D in D_range:

        print("D ", D)


        result_dict['demog'][D] = {}
        result_dict['demog'][D]['sojourn_time'] = []
        result_dict['demog'][D]['rescaled_diff_trajectories'] = []
        result_dict['demog'][D]['run_values'] = []
        result_dict['demog'][D]['slope'] = []

        while len(result_dict['demog'][D]['slope']) < n_iter:  

            x_trajectory_demog = simulate_demog_trajectory(m, r, D, n_samples, delta_t=1)
            diff_demog = (x_trajectory_demog - x_trajectory_demog[0])[1:]
            run_values_demog, run_starts_demog, run_lengths_demog = find_runs(diff_demog>0)

            run_sum_demog = numpy.asarray([sum(numpy.absolute(diff_demog[run_starts_demog[run_j_idx]:run_starts_demog[run_j_idx]+run_lengths_demog[run_j_idx]])) for run_j_idx in range(len(run_lengths_demog))])

            to_keep_idx = (run_lengths_demog >= min_run_length) & (run_sum_demog>0)
            run_values_demog_filter = run_values_demog[to_keep_idx]
            run_starts_demog_filter = run_starts_demog[to_keep_idx]
            run_length_demog_filter = run_lengths_demog[to_keep_idx]
            run_sum_demog_filter = run_sum_demog[to_keep_idx]

            # only fit regressions if there are at least 10 samples.
            if len(run_sum_demog_filter) < 10:
                continue

            slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_demog_filter), numpy.log10(run_sum_demog_filter))

            if (slope < 0) or (slope > 2):
                    continue

            for trajectory_i_idx in range(len(run_values_demog_filter)):

                run_starts_demog_filter_i = run_starts_demog_filter[trajectory_i_idx]
                run_length_demog_filter_i = run_length_demog_filter[trajectory_i_idx]
                run_values_demog_filter_i = run_values_demog_filter[trajectory_i_idx]

                trajectory_i = numpy.absolute(diff_demog[run_starts_demog_filter_i:(run_starts_demog_filter_i + run_length_demog_filter_i)])
                rescaled_trajectory_i = trajectory_i/(run_length_demog_filter_i**slope)
                result_dict['demog'][D]['rescaled_diff_trajectories'].append(rescaled_trajectory_i.tolist())

            # save all the run lengths so we get the full distribution...
            result_dict['demog'][D]['run_values'].extend(run_values_demog.tolist())
            result_dict['demog'][D]['sojourn_time'].extend(run_lengths_demog.tolist())
            result_dict['demog'][D]['slope'].append(slope)
    

    sys.stderr.write("Saving dictionary...\n")
    with open(sim_dict_path, 'wb') as outfile:
        pickle.dump(result_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")
    


def plot_universality():

    param_dict = pickle.load(open(sim_dict_path, "rb"))

    k = 1000
    tau = 20

    m = 100
    r = 0.1

    colors = ['powderblue', 'deepskyblue' , 'royalblue']


    fig = plt.figure(figsize = (8, 12.5)) #
    fig.subplots_adjust(bottom= 0.15)
    gs = gridspec.GridSpec(nrows=3, ncols=2)

    for model_idx, model in enumerate(['slm', 'demog']):

        ax_ex = fig.add_subplot(gs[0, model_idx])
        ax_sojourn_dist = fig.add_subplot(gs[1, model_idx])
        ax_dev_dist = fig.add_subplot(gs[2, model_idx])

        ax_ex.set_title(model, fontsize=12)

        for key_idx, key in enumerate(param_dict[model].keys()):
            
            param_dict_key = param_dict[model][key]

            if model == 'slm':
                #if key ==1:
                #    key = 1
                q_trajectory = simulate_slm_trajectory(1000, k, key, tau, delta_t=1)
                x_trajectory = numpy.exp(q_trajectory)
                label_ = r'$\sigma = $' + str(key)

            else:
                x_trajectory = simulate_demog_trajectory(m, r, key, 1000, delta_t=1)
                label_ = r'$D = $' + str(key)
            

            diff = x_trajectory - x_trajectory[0]
            ax_ex.plot(range(len(diff))[:500], diff[:500], lw=1, ls='-', alpha=0.7, color=colors[key_idx], label=label_)
            ax_ex.set_xlabel('Time, ' + r't', fontsize=10)
            ax_ex.set_ylabel('Deviation from origin, ' + r'x(t) - x(0)', fontsize=10)

            run_values = numpy.asarray(param_dict_key['run_values'])
            sojourn_time = numpy.asarray(param_dict_key['sojourn_time'])
            sojourn_time_pos = sojourn_time[run_values==True]
            sojourn_time_neg = sojourn_time[run_values==False]

            ax_sojourn_dist.hist(numpy.log10(sojourn_time_pos), 40, ls='-', histtype='step', stacked=True, fill=False, color=colors[key_idx], alpha=0.6)
            ax_sojourn_dist.hist(numpy.log10(sojourn_time_neg), 40, ls=':', histtype='step', stacked=True, fill=False, color=colors[key_idx], alpha=0.6)

            
            ax_sojourn_dist.set_yscale('log', base=10)
            ax_sojourn_dist.set_xlabel('Sojourn time, ' + r'T', fontsize=10)
            ax_sojourn_dist.set_ylabel("Probability density", fontsize=10)

            ax_sojourn_dist.set_xlim([0, 3.3])

            #print(numpy.std(param_dict_key['sojourn_time'])/numpy.mean(param_dict_key['sojourn_time']))
            #ax_sojourn_dist.set_xlim([0, 1100])


            # distribution
            rescaled_diff_trajectories_all = []
            rescaled_sojourn_time_all = []
            for i_idx in range(len(param_dict_key['rescaled_diff_trajectories'])):

                rescaled_diff_trajectories_all.extend(param_dict_key['rescaled_diff_trajectories'][i_idx])
                rescaled_sojourn_time_all.extend((numpy.arange(param_dict_key['sojourn_time'][i_idx])/(param_dict_key['sojourn_time'][i_idx]-1)).tolist())

                
            rescaled_diff_trajectories_all = numpy.asarray(rescaled_diff_trajectories_all)
            rescaled_sojourn_time_all = numpy.asarray(rescaled_sojourn_time_all)

            #bins_x_to_keep, bins_y = plot_utils.get_bin_x_mean_y(rescaled_sojourn_time_all, rescaled_diff_trajectories_all, bins=40, min_n_in_bin=5)
            
            #ax_dev_dist.plot(bins_x_to_keep, bins_y, ls='-', lw='2', color=colors[key_idx])
            #ax_dev_dist.set_xlabel('Time scaled by sojourn time, ' + r't/T', fontsize=10)
            #ax_dev_dist.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right > \cdot T^{-\alpha}$', fontsize=10)


        ax_ex.legend(loc='upper right')


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%suniversality.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()






#plot_sojourn_time_vs_cumulative_walk_length()


#plot_sojourn_time_vs_cumulative_walk_length_demog(100, 0.1, 1, 10000)

#plot_sojourn_time_vs_cumulative_walk_length_slm(K = 1000, sigma = 1, tau = 7)


#simulate_sojourn_time_vs_cumulative_walk_length(n_samples=10000, n_iter=100)

#plot_one_slm_trajectory()

 
#plot_slope_sim()


#plot_dist_run_lengths()
#simulate_universality(n_samples=10000, n_iter=100, min_run_length=10)


#plot_universality()