
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


environment = 'gut'
occupancy_min = 1.0


ax_idx_all = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1), (4,0), (4,1)]

slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sim_dict_path = '%sfluctuation_sim_dict.pickle' %config.data_directory


def make_null_exponent_dict(n_iter=1, min_run_length=data_utils.min_run_length_data, epsilon_fract=data_utils.epsilon_fract_data):

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    exponent_null_all = []

    for n in range(n_iter):

        for dataset in data_utils.dataset_all:

            for host in mle_dict[dataset].keys():

                for asv in mle_dict[dataset][host]:

                    rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                    x_mean = mle_dict[dataset][host][asv]['x_mean']
                    days = numpy.asarray(mle_dict[dataset][host][asv]['days'])

                    rel_abundance_null = numpy.random.permutation(rel_abundance)

                    #run_values_null, run_starts_null, run_lengths_null = find_runs((rel_abundance_null - x_mean)>0, min_run_length=1)

                    run_dict = data_utils.calculate_deviation_pattern_data(rel_abundance_null, x_mean, days, min_run_length=min_run_length, epsilon=epsilon_fract, return_array=False)

                    print(run_dict)


def plot_sojourn_vs_norm(remove_negative_values=True):

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    sojourn_time_all = []
    norm_all = []

    fig, ax = plt.subplots(figsize=(6,4))

    for dataset in data_utils.dataset_all:

        for host in mle_dict[dataset].keys():

            sojourn_time_all_host = []
            norm_all_host = []

            for asv in mle_dict[dataset][host]:

                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict == None:
                    continue

                for sojourn_time, sojourn_trajectory_list_nested in run_dict.items():

                    for sojourn_trajectory_list in sojourn_trajectory_list_nested:

                        # len(sojourn_trajectory_list) is the number of observations in the sojourn
                        # sojourn_time is the number of days

                        rescald_time_range = numpy.linspace(0, 1, num=len(sojourn_trajectory_list), endpoint=True)

                        rescaled_sojourn_trajectory_list = numpy.asarray(sojourn_trajectory_list)/mle_dict[dataset][host][asv]['x_mean']

                        #to_keep_idx = sojourn_trajectory_list > 0

                        #sojourn_trajectory_list = sojourn_trajectory_list[to_keep_idx]
                        #rescald_time_range_ = rescald_time_range[to_keep_idx]

                        #print(to_keep_idx[0], to_keep_idx[-1], len(to_keep_idx))
                        #print(numpy.log(sojourn_trajectory_list))

                        # we are taking the integral of the *log*
                        # add one because we want the constant to be positive and to *increase* with T
                        print(mle_dict[dataset][host][asv]['x_mean'])
                        print(sojourn_trajectory_list)
                        print(rescaled_sojourn_trajectory_list)
                        rescaled_sojourn_trajectory_log = numpy.log(rescaled_sojourn_trajectory_list+1)
                        print(rescaled_sojourn_trajectory_log)
                        #rescald_time_range_ = numpy.linspace(0, 1, num=len(rescald_time_range_), endpoint=True)
                        norm = stats_utils.estimate_normalization_constant(rescald_time_range, rescaled_sojourn_trajectory_log)

                        print(norm)

                        sojourn_time_all.append(sojourn_time)
                        norm_all.append(norm)
            
                        sojourn_time_all_host.append(sojourn_time)
                        norm_all_host.append(norm)

            
            ax.scatter(sojourn_time_all_host, norm_all_host, color=plot_utils.host_color_dict[dataset][host], alpha=0.6, s=10)


    sojourn_time_all = numpy.asarray(sojourn_time_all)
    norm_all = numpy.asarray(norm_all)
    slope, intercept, r_value, p_value, std_err = stats_utils.log_log_regression(sojourn_time_all, norm_all)
    print(slope, p_value, std_err)

    x_log10_range =  numpy.linspace(min(numpy.log10(sojourn_time_all)) , max(numpy.log10(sojourn_time_all)) , 10000)
    y_log10_fit_range = (slope*x_log10_range + intercept)

    ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")

    ax.text(0.5, 0.1, 'Exponent = %.2f' % slope, fontsize=13, transform=ax.transAxes)


    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel('Sojourn time, ' + r'$T$', fontsize=12)
    ax.set_ylabel("Normalization constant, " + r'$N(T)$', fontsize=12)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_vs_norm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_deviation_data(sojourn_time, min_x_mean = 0):

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    x_mean_all = []
    for dataset in data_utils.dataset_all:
        for host in mle_dict[dataset].keys():
            for asv in  mle_dict[dataset][host].keys():
                
                if mle_dict[dataset][host][asv]['x_mean'] <= min_x_mean:
                    continue

                x_mean_all.append(mle_dict[dataset][host][asv]['x_mean'])

    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(x_mean_all), max(x_mean_all)), cmap = plt.get_cmap('Blues'))


    run_length_all = []

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            host_count+=1

            ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=9)
            ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=9)


            for asv in  mle_dict[dataset][host].keys():
                
                rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                x_mean = mle_dict[dataset][host][asv]['x_mean']
                #cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_mean']

                if x_mean <= min_x_mean:
                    continue

                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict is None:
                    continue

                run_length_all.extend(list(run_dict.keys()))

                for run_length, run_sojourn in run_dict.items():

                    if run_length != sojourn_time:
                        continue

                    s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)
 
                    for run_sojourn_j in run_sojourn:

                        run_sojourn_j = numpy.asarray(run_sojourn_j)

                        run_sojourn_integral_j = integrate.simpson(run_sojourn_j, s_range)
                        run_sojourn_j = run_sojourn_j/run_sojourn_integral_j

                        if x_mean < 0.001:
                            continue
                        

                        #print(len(run_sojourn_j))
                        #run_sojourn_j = numpy.absolute(run_sojourn_j)
                        
                        # ensure no negative values
                        to_plot_idx = (run_sojourn_j>0)
                        s_range_to_plot = s_range[to_plot_idx]
                        run_sojourn_j_to_plot = run_sojourn_j[to_plot_idx]
                        #rescaled_run_sojourn_j_to_plot = run_sojourn_j_to_plot/x_mean
                        ax.plot(s_range_to_plot, run_sojourn_j_to_plot, lw=1, alpha=1, c=cmap.to_rgba(x_mean), ls='-')

                
            ax.set_yscale('log', base=10)
            ax.set_xlim([0,1])


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdeviation_data_rescaled/deviation_data_rescaled_%d.png" % (config.analysis_directory, sojourn_time)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()


def plot_deviation_data_all_sojourn(min_x_mean = 0):


    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    host_count = 0

    cmap = cm.ScalarMappable(norm = colors.Normalize(11, 20), cmap = plt.get_cmap('Blues'))


    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            days_run_lengths_all = []
            cv_all = []
            for key, value in mle_dict[dataset][host].items():

                days_run_lengths_all.extend(value['days_run_lengths'])

                cv_all.append(value['x_std']/value['x_mean'])

                
            days_run_lengths_all = numpy.asarray(days_run_lengths_all)

            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])

            sojourn_dict = {}

            host_count+=1

            for asv in  mle_dict[dataset][host].keys():
                
                rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                x_mean = mle_dict[dataset][host][asv]['x_mean']

                if x_mean <= min_x_mean:
                    continue

                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict is None:
                    continue

                for run_length, run_sojourn in run_dict.items():

                    #s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)

                    if run_length not in sojourn_dict:
                        sojourn_dict[run_length] = []
 
                    for run_sojourn_j in run_sojourn:

                        run_sojourn_j = numpy.asarray(run_sojourn_j)
                        run_sojourn_rescaled_j = run_sojourn_j/x_mean

                        #run_sojourn_integral_j = integrate.simpson(run_sojourn_j, s_range)
                        #run_sojourn_j = run_sojourn_j/run_sojourn_integral_j

                        sojourn_dict[run_length].append(run_sojourn_rescaled_j)

                        # ensure no negative values
                        #to_plot_idx = (run_sojourn_j>0)
                        #s_range_to_plot = s_range[to_plot_idx]
                        #run_sojourn_j_to_plot = run_sojourn_j[to_plot_idx]


            mean_run_sojourn_all = []
            for run_length, run_sojourn_all in sojourn_dict.items():

                if len(run_sojourn_all) <= 4:
                    continue
                

                run_sojourn_all = numpy.stack(run_sojourn_all, axis=0)
                mean_run_sojourn = numpy.mean(run_sojourn_all, axis=0)

                s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)

                #print(mean_run_sojourn[-1])
                to_plot_idx = (mean_run_sojourn>0)
                #s_range_to_plot = s_range[to_plot_idx]
                #mean_run_sojourn_to_plot = mean_run_sojourn[to_plot_idx]

                ax.plot(s_range, mean_run_sojourn, lw=1, alpha=1, c=cmap.to_rgba(run_length), ls='-')

                mean_run_sojourn_all.extend(mean_run_sojourn.tolist())

            #ax.set_yscale('log', base=10)
            ax.set_xlim([0,1])
            ax.set_ylim([0,max(mean_run_sojourn_all)*1.1])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            #host_count+=1
            ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=9)
            ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=9)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdeviation_data_all_sojourn.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_deviation_data_all_sojourn_survival(min_x_mean = 0):

    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    host_count = 0

    cmap = cm.ScalarMappable(norm = colors.Normalize(11, 20), cmap = plt.get_cmap('Blues'))


    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        ax = plt.subplot2grid((1, 3), (0, dataset_idx))

        sojourn_deviation_all = []

        for host in host_all:

            days_run_lengths_all = []
            cv_all = []
            for key, value in mle_dict[dataset][host].items():

                days_run_lengths_all.extend(value['days_run_lengths'])

                cv_all.append(value['x_std']/value['x_mean'])

                
            days_run_lengths_all = numpy.asarray(days_run_lengths_all)

    
            sojourn_dict = {}

            for asv in  mle_dict[dataset][host].keys():
                
                rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                x_mean = mle_dict[dataset][host][asv]['x_mean']

                if x_mean <= min_x_mean:
                    continue

                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict is None:
                    continue

                for run_length, run_sojourn in run_dict.items():

                    #s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)

                    if run_length not in sojourn_dict:
                        sojourn_dict[run_length] = []
 
                    for run_sojourn_j in run_sojourn:

                        run_sojourn_j = numpy.asarray(run_sojourn_j)
                        run_sojourn_rescaled_j = run_sojourn_j/x_mean
                        sojourn_dict[run_length].append(run_sojourn_rescaled_j)


            mean_run_sojourn_all = []
            for run_length, run_sojourn_all in sojourn_dict.items():

                #if len(run_sojourn_all) <= 4:
                #    continue

                for run_sojourn_j in run_sojourn_all:
                    sojourn_deviation_all.extend(run_sojourn_j.tolist())


        sojourn_deviation_all = numpy.asarray(sojourn_deviation_all)
        sojourn_deviation_all = sojourn_deviation_all[sojourn_deviation_all>0]

        

        

        print(mean_run_sojourn)

    ax.plot(s_range, mean_run_sojourn, lw=1, alpha=1, c=cmap.to_rgba(run_length), ls='-')

    mean_run_sojourn_all.extend(mean_run_sojourn.tolist())

    #ax.set_yscale('log', base=10)
    ax.set_xlim([0,1])
    ax.set_ylim([0,max(mean_run_sojourn_all)*1.1])
    ax.set_title('%s, %s' % (dataset, host), fontsize=11)
    #host_count+=1
    ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=9)
    ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=9)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdeviation_data_all_sojourn_survival.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_dist_run_lengths():

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    mle_dict = pickle.load(open(mle_dict_path, "rb"))

    host_count = 0

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            days_run_lengths_all = []
            cv_all = []
            for key, value in mle_dict[dataset][host].items():

                days_run_lengths_all.extend(value['days_run_lengths'])

                cv_all.append(value['x_std']/value['x_mean'])

                
            days_run_lengths_all = numpy.asarray(days_run_lengths_all)

            #print(dataset, host, numpy.mean(cv_all))


            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            #days_run_lengths_all_pos = days_run_lengths_all[run_values_all==True]
            #days_run_lengths_all_neg = days_run_lengths_all[run_values_all==False]

            days_run_lengths_all_log10 = numpy.log10(days_run_lengths_all)
            hist_to_plot, bins_mean_to_plot = data_utils.get_hist_and_bins(days_run_lengths_all_log10, n_bins=10)
            #ax.hist(numpy.log10(days_run_lengths_all), bins=9, density=True, lw=3, histtype='step', fill=False, color='dodgerblue')
            ax.scatter(10**bins_mean_to_plot, hist_to_plot, s=40, color='dodgerblue', alpha=1, lw=1)

            #ax.hist(days_run_lengths_all_pos, bins=10, density=True, histtype='step', fill=False, color='b')
            #ax.hist(days_run_lengths_all_neg, bins=10, density=True, histtype='step', fill=False, color='r')

            ax.set_xlabel("Sojourn time (days)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            
            ax.set_xlim([1, 200])
            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)
            #

            #ax.set_xlim([0, numpy.log10(200)])


            host_count+=1


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdist_run_lengths.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_stat_vs_sojourn_data():

    fig = plt.figure(figsize = (12, 12)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    host_count = 0

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        ax_mean = plt.subplot2grid((3, len(data_utils.dataset_all)), (0, dataset_idx))
        ax_cv = plt.subplot2grid((3, len(data_utils.dataset_all)), (1, dataset_idx))
        ax_cv_resid = plt.subplot2grid((3, len(data_utils.dataset_all)), (2, dataset_idx))

        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        mean_all_all = []
        cv_all_all = []
        mean_sojourn_all_all = []
        for host in host_all:

            mean_all = []
            cv_all = []
            mean_sojourn_all = []
            for key, value in mle_dict[dataset][host].items():

                mean_all.append(value['x_mean'])
                cv_all.append(value['x_std']/value['x_mean'])
                mean_sojourn_all.append(numpy.mean(value['days_run_lengths']))

            
            ax_mean.scatter(mean_all, mean_sojourn_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.host_color_dict[dataset][host])
            ax_cv.scatter(cv_all, mean_sojourn_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.host_color_dict[dataset][host])

            mean_all_all.extend(mean_all)
            cv_all_all.extend(cv_all)
            mean_sojourn_all_all.extend(mean_sojourn_all)
            
        
        #if dataset_idx < 2:

        bins_mean, bins_mean_sojourn = plot_utils.get_bin_mean_x_y(mean_all_all, mean_sojourn_all_all, bins=15, min_n_bin=3)
        bins_cv, bins_cv_sojourn = plot_utils.get_bin_mean_x_y(cv_all_all, mean_sojourn_all_all, bins=15, min_n_bin=3)
        
        #ax_mean.plot(bins_mean, bins_mean_sojourn, c='k', lw=3, linestyle='--', zorder=2)
        #ax_cv.plot(bins_cv, bins_cv_sojourn, c='k', lw=3, linestyle='--', zorder=2)

        ax_mean.set_title('%s' % (dataset), fontsize=12)



        # slope
        slope_mean, intercept_mean, r_value_mean, p_value_mean, std_err_mean = data_utils.stats.linregress(numpy.log10(mean_all_all), numpy.log10(mean_sojourn_all_all))
        slope_cv, intercept_cv, r_value_cv, p_value_cv, std_err_cv = data_utils.stats.linregress(numpy.log10(cv_all_all), numpy.log10(mean_sojourn_all_all))

        x_log10_range_mean =  numpy.linspace(min(numpy.log10(mean_all_all)) , max(numpy.log10(mean_all_all)) , 10000)
        y_log10_fit_range_mean = (slope_mean*x_log10_range_mean + intercept_mean)

        x_log10_range_cv =  numpy.linspace(min(numpy.log10(cv_all_all)) , max(numpy.log10(cv_all_all)) , 10000)
        y_log10_fit_range_cv = (slope_cv*x_log10_range_cv + intercept_cv)

        #if p_value_mean < 0.05:
        #    ax_mean.plot(10**x_log10_range_mean, 10**y_log10_fit_range_mean, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
        
        ax_mean.plot(10**x_log10_range_mean, 10**y_log10_fit_range_mean, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
        ax_cv.plot(10**x_log10_range_cv, 10**y_log10_fit_range_cv, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")

        #if p_value_cv < 0.05:
        #    ax_cv.plot(10**x_log10_range_cv, 10**y_log10_fit_range_cv, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")

        cv_resid = numpy.log10(mean_sojourn_all_all) - (slope_cv*numpy.log10(cv_all_all) + intercept_cv)

        ax_cv_resid.scatter(cv_all_all, cv_resid, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.host_color_dict[dataset][host])
        #ax_mean.scatter(mean_all_all, mean_sojourn_all, s=10, alpha=0.8, c=plot_utils.host_color_dict[dataset][host], zorder=1)

        slope_resid, intercept_resid, r_value_resid, p_value_resid, std_err_resid = data_utils.stats.linregress(numpy.log10(cv_all_all), cv_resid)
        print(slope_cv, p_value_cv)
        y_log10_fit_range_cv_resid = (slope_resid*x_log10_range_cv + intercept_resid)
        ax_cv_resid.plot(10**x_log10_range_cv, y_log10_fit_range_cv_resid, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")

        ax_mean.set_xscale('log', base=10)
        ax_cv.set_xscale('log', base=10)

        ax_mean.set_yscale('log', base=10)
        ax_cv.set_yscale('log', base=10)

        ax_mean.set_xlabel("Mean abundance", fontsize=12)
        ax_cv.set_xlabel("CV of abundance", fontsize=12)

        ax_mean.set_ylabel("Mean sojourn time", fontsize=12)
        ax_cv.set_ylabel("Mean sojourn time", fontsize=12)


        ax_cv_resid.set_xlabel("CV of abundance", fontsize=12)
        ax_cv_resid.set_ylabel("Residuals", fontsize=12)
        ax_cv_resid.axhline(y=0, lw=2, ls=':', c='k')

        ax_cv_resid.set_xscale('log', base=10)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sstat_vs_mean_sojourn_data.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    #plot_sojourn_vs_norm()

    #plot_deviation_data_all_sojourn_survival()

    #plot_stat_vs_sojourn_data()

    #plot_dist_run_lengths()
    
    #plot_sojourn_vs_norm()
    plot_stat_vs_sojourn_data()
    #plot_deviation_data(11)

    #plot_stat_vs_sojourn_data()
    #make_mle_dict(epsilon_fract=0.01)

    #plot_deviation_data_all_sojourn()

    #make_null_exponent_dict()


#plot_deviation_data(12)
