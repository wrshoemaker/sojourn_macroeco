import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
import scipy.stats as stats
from scipy import integrate
from scipy.spatial import distance
from scipy.special import digamma, gamma


import data_utils
import plot_utils
import stats_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import simulation_utils

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)



def plot_compare_slopes():

    sojourn_dict = pickle.load(open('%ssojourn_time_dist_sim_fixed_moments_bdm_dict.pickle' % config.data_directory, 'rb'))

    mean = list(sojourn_dict.keys())[0]
    cv_all = list(sojourn_dict[mean].keys())
    tau_all = list(sojourn_dict[mean][cv_all[0]].keys())

    slopes_bdm = []
    slopes_slm = []

    cv_all_to_plot = []
    tau_all_to_plot = []
    for cv in cv_all:

        for tau in tau_all:

            sojourn_dict_bdm_i = sojourn_dict[mean][cv][tau]['bdm']['sojourn_time_count_dict']
            sojourn_dict_slm_i = sojourn_dict[mean][cv][tau]['slm']['sojourn_time_count_dict']

            #print(sojourn_dict[mean][cv][tau]['bdm']['slope'])

            x_bdm = numpy.asarray(list(sojourn_dict_bdm_i.keys()))
            x_slm = numpy.asarray(list(sojourn_dict_slm_i.keys()))

            if (max(x_bdm) > 400) or (max(x_slm) > 400):
                continue

            if sojourn_dict[mean][cv][tau]['bdm']['slope'] < 0:
                print(cv, tau)

            #print(max(x_bdm), max(x_slm))

            slopes_bdm.append(sojourn_dict[mean][cv][tau]['bdm']['slope'])
            slopes_slm.append(sojourn_dict[mean][cv][tau]['slm']['slope'])
            cv_all_to_plot.append(cv)
            tau_all_to_plot.append(tau)


    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_cv = plt.subplot2grid((1,2), (0,0))
    ax_timescale = plt.subplot2grid((1,2), (0,1))

    ax_cv.scatter(cv_all_to_plot, slopes_bdm, s=10, c='r', alpha=0.2, label='BDM')
    ax_cv.scatter(cv_all_to_plot, slopes_slm, s=10, c='b', alpha=0.2, label='SLM')
    ax_cv.set_xlabel("CV", fontsize=12)
    ax_cv.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_cv.set_xscale('log', base=10)
    ax_cv.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')
    ax_cv.axhline(y=0.1531465459645992, lw=2, ls='--', c='k', label='Human gut')
    ax_cv.legend(loc='upper left')


    ax_timescale.scatter(tau_all_to_plot, slopes_bdm, s=10, c='r', alpha=0.2)
    ax_timescale.scatter(tau_all_to_plot, slopes_slm, s=10, c='b', alpha=0.2)
    ax_timescale.set_xlabel("Growth timescale", fontsize=12)
    ax_timescale.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_timescale.set_xscale('log', base=10)
    ax_timescale.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')
    ax_timescale.axhline(y=0.1531465459645992, lw=2, ls='--', c='k', label='Human gut')


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scv_vs_slope_slm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





def plot_compare_slopes_old():

    demog_dict = pickle.load(open(slm_dict_path, "rb"))

    slopes_linear = []
    slopes_sqrt = []

    cv_x_all = []
    mean_x_all = []
    timescale_all = []
    for sigma in demog_dict.keys():

        for k in demog_dict[sigma].keys():

            for tau in demog_dict[sigma][k].keys():
                
                slopes_linear.append(demog_dict[sigma][k][tau]['linear']['slope'])
                slopes_sqrt.append(demog_dict[sigma][k][tau]['log']['slope'])

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k, sigma)

                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)
                timescale_all.append(tau)


    cv_x_all = numpy.asarray(cv_x_all)
    mean_x_all = numpy.asarray(mean_x_all)
    fractile_cv = sum(cv_x_all>3)/len(cv_x_all)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(slopes_linear, slopes_sqrt, s=10, c='k', alpha=0.2)

    min_ = min(slopes_linear+slopes_sqrt)
    max_ = max(slopes_linear+slopes_sqrt)

    ax.plot([min_, max_], [min_, max_], ls=':', lw=2, c='k')

    ax.set_xlabel('Slope, linear', fontsize=12)
    ax.set_ylabel("Slope, log", fontsize=12)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scompare_slopes_sqrt.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()


    # investigate problematic parameter regimes
    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_mean = plt.subplot2grid((1,3), (0,0))
    ax_cv = plt.subplot2grid((1,3), (0,1))
    ax_timescale = plt.subplot2grid((1,3), (0,2))

    ax_mean.scatter(mean_x_all, slopes_linear, s=10, c='b', alpha=0.2, label='No transformation')
    ax_mean.scatter(mean_x_all, slopes_sqrt, s=10, c='r', alpha=0.2, label = 'Square-root transformation')
    ax_mean.set_xlabel("Mean", fontsize=12)
    ax_mean.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_mean.set_xscale('log', base=10)
    ax_mean.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')
    ax_mean.legend(loc='upper right')

    ax_cv.scatter(cv_x_all, slopes_linear, s=10, c='b', alpha=0.2)
    ax_cv.scatter(cv_x_all, slopes_sqrt, s=10, c='r', alpha=0.2)
    ax_cv.set_xlabel("CV", fontsize=12)
    ax_cv.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_cv.set_xscale('log', base=10)
    ax_cv.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')


    ax_timescale.scatter(timescale_all, slopes_linear, s=10, c='b', alpha=0.2)
    ax_timescale.scatter(timescale_all, slopes_sqrt, s=10, c='r', alpha=0.2)
    ax_timescale.set_xlabel("Autocorr. timescale (1/birth)", fontsize=12)
    ax_timescale.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_timescale.set_xscale('log', base=10)
    ax_timescale.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scv_vs_slope_slm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





def plot_rescaled_deviation_all_params():

    demog_dict = pickle.load(open(slm_dict_path, "rb"))
    
    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    for data_type_idx, data_type in enumerate(['linear', 'log']):

        ax = plt.subplot2grid((1,2), (0,data_type_idx))
        ax.set_xlabel('Rescaled time within sojourn period', fontsize=12)
        ax.set_ylabel("Rescaled mean deviation", fontsize=12)
        ax.set_title(data_type, fontsize=12)

        for sigma in demog_dict.keys():

            for k in demog_dict[sigma].keys():

                for tau in demog_dict[sigma][k].keys():
                
                    run_length = demog_dict[sigma][k][tau][data_type]['run_length']
                    mean_run_deviation = demog_dict[sigma][k][tau][data_type]['mean_run_deviation']

                    intercept = demog_dict[sigma][k][tau][data_type]['intercept']
                    norm_constant = demog_dict[sigma][k][tau][data_type]['norm_constant']

                    s_range_all = []
                    rescaled_mean_run_deviation_all = []

                    for run_length_i_idx, run_length_i in enumerate(run_length):

                        s_range_i = numpy.linspace(0, 1, num=run_length_i, endpoint=True)
                        #rescaled_mean_run_deviation_i = mean_run_deviation[run_length_i_idx]/(norm_constant[run_length_i_idx]/(10**intercept))

                        s_range_all.extend(s_range_i.tolist())
                        #rescaled_mean_run_deviation_all.extend(rescaled_mean_run_deviation_i.tolist())
                        rescaled_mean_run_deviation_all.extend(mean_run_deviation[run_length_i_idx])

                    
                    s_range_all = numpy.asarray(s_range_all)
                    rescaled_mean_run_deviation_all = numpy.asarray(rescaled_mean_run_deviation_all)

                    hist_x_all, bin_edges_x_all = numpy.histogram(s_range_all, density=True, bins=50)
                    #bins_x_all = [0.5 * (bin_edges_x_all[i] + bin_edges_x_all[i+1]) for i in range(0, len(bin_edges_x_all)-1 )]
                    bins_x_all_to_keep = []
                    bins_y = []
                    for i in range(0, len(bin_edges_x_all)-1 ):
                        y_i = rescaled_mean_run_deviation_all[(s_range_all>=bin_edges_x_all[i]) & (s_range_all<bin_edges_x_all[i+1])]

                        if len(y_i) < 10:
                            continue
                       
                        bins_x_all_to_keep.append(bin_edges_x_all[i])
                        bins_y.append(numpy.mean(y_i))


                    if len(bins_x_all_to_keep) < 10:
                        continue
                    
                    print(bins_x_all_to_keep)
                    if (bins_x_all_to_keep[0] != float(0)) or (bins_x_all_to_keep[0] != float(1)):
                        continue


                    bins_x_all_to_keep = numpy.asarray(bins_x_all_to_keep)
                    bins_y = numpy.asarray(bins_y)

                    ax.plot(bins_x_all_to_keep, bins_y, lw=0.5, ls='-', c='k', alpha=0.3)

                    #if (data_type == 'sqrt') and (max(bins_y) > 8.5):

                    #    print(m,r,D)

                    #if (data_type == 'linear') and (max(bins_y) > 40000):

                    #    print(m,r,D)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_deviation_all_params_slm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_rescaled_same_sojourn_time(sojourn_time=25, delta_t=1):

    slm_dict = pickle.load(open(slm_dict_path, "rb"))

    s_range = numpy.linspace(0, 1, num=sojourn_time, endpoint=True)

    mean_x_all = []
    cv_x_all = []
    timescale_x_all = []
    tau_all = []
    noise_constant_all = [] 
    for sigma in slm_dict.keys():

        for k in slm_dict[sigma].keys():

            for tau in slm_dict[sigma][k].keys():

                if (sojourn_time not in (slm_dict[sigma][k][tau]['linear']['run_length'])) or (sojourn_time not in (slm_dict[sigma][k][tau]['log']['run_length'])):
                        continue

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)
                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)
                timescale_x_all.append(1/tau)
                tau_all.append(tau)

                noise_constant_all.append(numpy.sqrt(sigma*delta_t/tau))



    #rgb_blue = cm.Blues(numpy.logspace(0, 1, max(run_length_all), base=10))

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
    fig.suptitle('Sojourn time ' + r'$T=$' + str(sojourn_time), fontsize=12)

    tau_all = numpy.asarray(tau_all)
    noise_constant_all = numpy.asarray(noise_constant_all)
    #cmap = cm.ScalarMappable(norm = colors.LogNorm(min(noise_constant_all), max(noise_constant_all)), cmap = plt.get_cmap('Blues'))
    cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))


    for data_type_idx, data_type in enumerate(['linear', 'log']):

        ax = plt.subplot2grid((1,2), (0,data_type_idx))
        ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=12)
        ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=12)
        ax.set_title(plot_utils.data_type_title_dict[data_type], fontsize=12)
        #ax.set_yscale('log', base=10)

        for sigma in slm_dict.keys():

            for k in slm_dict[sigma].keys():

                for tau in slm_dict[sigma][k].keys():

                    if (sojourn_time not in (slm_dict[sigma][k][tau]['linear']['run_length'])) or (sojourn_time not in (slm_dict[sigma][k][tau]['log']['run_length'])):
                        continue

                    run_length = slm_dict[sigma][k][tau][data_type]['run_length']
                    mean_run_deviation = slm_dict[sigma][k][tau][data_type]['mean_run_deviation']

                    intercept = slm_dict[sigma][k][tau][data_type]['intercept']
                    norm_constant = slm_dict[sigma][k][tau][data_type]['norm_constant']

                    run_length = numpy.asarray(run_length)
                    sojourn_time_idx = numpy.where(run_length == sojourn_time)[0][0]

                    mean_run_deviation_target = numpy.asarray(mean_run_deviation[sojourn_time_idx])

                    mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)
                    mean_run_deviation_target_rescaled_by_mean = mean_run_deviation_target/mean_x

                    noise_term = numpy.sqrt(sigma*delta_t/tau)

                    ax.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', c=cmap.to_rgba(cv_x), alpha=0.3)

                    #rescaled_mean_run_deviation_all.extend(rescaled_mean_run_deviation_i.tolist())

        #if data_type == 'linear':
        ax.set_yscale('log', base=10)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_same_sojourn_time_slm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()









if __name__ == "__main__":


    print("Running...")

    #plot_compare_slopes()

    #plot_rescaled_deviation_all_params()

    #plot_rescaled_same_sojourn_time()

    plot_compare_slopes()





