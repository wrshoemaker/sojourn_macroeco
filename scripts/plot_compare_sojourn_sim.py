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

import simulation_utils



#demog_dict = pickle.load(open(simulation_utils.demog_dict_path, "rb"))
dist_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_fixed_moments_dict_path, "rb"))


data_type = 'linear'

sojourn_time=15
delta_t=1

fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

ax_bdm = plt.subplot2grid((1,2), (0,0))
ax_slm = plt.subplot2grid((1,2), (0,1))

s_range = numpy.linspace(0, 1, num=sojourn_time, endpoint=True)



def old_plot():

    # get all cv to make colormap
    mean_x_all = []
    cv_x_all = []
    for m in demog_dict.keys():

        for r in demog_dict[m].keys():

            for D in demog_dict[m][r].keys():

                if (sojourn_time not in (demog_dict[m][r][D]['linear']['run_length'])) or (sojourn_time not in (demog_dict[m][r][D]['sqrt']['run_length'])):
                    continue
                    
                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)
                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)


    for sigma in slm_dict.keys():

        for k in slm_dict[sigma].keys():

            for tau in slm_dict[sigma][k].keys():

                if (sojourn_time not in (slm_dict[sigma][k][tau]['linear']['run_length'])) or (sojourn_time not in (slm_dict[sigma][k][tau]['log']['run_length'])):
                    continue

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)
                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)


    mean_x_all = numpy.asarray(mean_x_all)
    cv_x_all = numpy.asarray(cv_x_all)

    #cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))
    #cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))
    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))


    for m in demog_dict.keys():

        for r in demog_dict[m].keys():

            for D in demog_dict[m][r].keys():

                if (sojourn_time not in (demog_dict[m][r][D]['linear']['run_length'])) or (sojourn_time not in (demog_dict[m][r][D]['sqrt']['run_length'])):
                    continue

                run_length = demog_dict[m][r][D][data_type]['run_length']
                mean_run_deviation = demog_dict[m][r][D][data_type]['mean_run_deviation']

                intercept = demog_dict[m][r][D][data_type]['intercept']
                norm_constant = demog_dict[m][r][D][data_type]['norm_constant']

                run_length = numpy.asarray(run_length)
                sojourn_time_idx = numpy.where(run_length == sojourn_time)[0][0]

                mean_run_deviation_target = numpy.asarray(mean_run_deviation[sojourn_time_idx])

                print(mean_run_deviation_target)

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)
                mean_run_deviation_target_rescaled_by_mean = mean_run_deviation_target/mean_x

                ax_demog.plot(s_range, mean_run_deviation_target, lw=0.5, ls='-', c=cmap.to_rgba(cv_x), alpha=0.3)
                #ax_demog.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)



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

                ax_slm.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', c=cmap.to_rgba(cv_x), alpha=0.3)
                #ax_slm.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)


    sim_types = ['demog', 'slm']

    for ax_idx, ax in enumerate([ax_demog, ax_slm]):

        ax.set_yscale('log', base=10)
        ax.set_xlim([0,1])

        ax.set_xlabel('Relative time within sojourn period, ' + r'$t$', fontsize=12)
        ax.set_ylabel("Mean deviation scaled by stationary mean, " + r'$\frac{\left < x(t) - x(0) \right >_{T}}{\left < x \right >}$', fontsize=10)
        ax.set_title(plot_utils.sim_type_label_dict[sim_types[ax_idx]], fontsize=14)


    fig.subplots_adjust(hspace=0.25, wspace=0.35)
    fig_name = "%scompare_sojourn_sim.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()






mean = list(dist_dict.keys())[0]
cv_all = list(dist_dict[mean].keys())
tau_all = list(dist_dict[mean][cv_all[0]].keys())


print(tau_all)
cmap = cm.ScalarMappable(norm = colors.LogNorm(min(tau_all), max(tau_all)), cmap = plt.get_cmap('Blues'))

#cv_all = [cv_all[4]]

#cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))
#cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))


for cv in cv_all:

    for tau in tau_all:

        run_length_bdm = dist_dict[mean][cv][tau]['bdm']['run_length']
        run_length_slm = dist_dict[mean][cv][tau]['slm']['run_length']

        mean_run_deviation_bdm = dist_dict[mean][cv][tau]['bdm']['mean_run_deviation']
        mean_run_deviation_slm = dist_dict[mean][cv][tau]['slm']['mean_run_deviation']

        slope_bdm = dist_dict[mean][cv][tau]['bdm']['slope']
        slope_slm = dist_dict[mean][cv][tau]['slm']['slope']

        intercept_bdm = dist_dict[mean][cv][tau]['bdm']['intercept']
        intercept_slm = dist_dict[mean][cv][tau]['slm']['intercept']

        norm_constant_bdm = dist_dict[mean][cv][tau]['bdm']['norm_constant']
        norm_constant_slm = dist_dict[mean][cv][tau]['slm']['norm_constant']
        
        #print(dist_dict[mean][cv][tau]['slm'].keys)

        #intercept = demog_dict[m][r][D][data_type]['intercept']
        #norm_constant = demog_dict[m][r][D][data_type]['norm_constant']

        run_length_bdm = numpy.asarray(run_length_bdm)
        run_length_slm = numpy.asarray(run_length_slm)

        sojourn_time_bdm_idx = numpy.where(run_length_bdm == sojourn_time)[0][0]
        sojourn_time_slm_idx = numpy.where(run_length_slm == sojourn_time)[0][0]

        mean_run_deviation_target_bdm = numpy.asarray(mean_run_deviation_bdm[sojourn_time_bdm_idx])
        mean_run_deviation_target_slm = numpy.asarray(mean_run_deviation_slm[sojourn_time_slm_idx])

        #mean_run_deviation_target_bdm = mean_run_deviation_target_bdm/mean
        #mean_run_deviation_target_slm = mean_run_deviation_target_slm/mean

        mean_run_deviation_target_bdm = mean_run_deviation_target_bdm/norm_constant_bdm[sojourn_time_bdm_idx]
        mean_run_deviation_target_slm = mean_run_deviation_target_slm/norm_constant_slm[sojourn_time_slm_idx]

        to_plot_bdm_idx = (mean_run_deviation_target_bdm>0)
        to_plot_slm_idx = (mean_run_deviation_target_bdm>0)

        #print(mean_run_deviation_target_bdm[to_plot_bdm_idx])

        ax_bdm.plot(s_range[to_plot_bdm_idx], mean_run_deviation_target_bdm[to_plot_bdm_idx], lw=0.5, ls='-', c=cmap.to_rgba(tau), alpha=0.3)
        #ax_demog.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)

        ax_slm.plot(s_range[to_plot_bdm_idx], mean_run_deviation_target_slm[to_plot_bdm_idx], lw=0.5, ls='-', c=cmap.to_rgba(tau), alpha=0.3)
        #ax_slm.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)


sim_types = ['bdm', 'slm']

for ax_idx, ax in enumerate([ax_bdm, ax_slm]):

    ax.set_yscale('log', base=10)
    ax.set_xlim([0,1])

    ax.set_xlabel('Relative time within sojourn period, ' + r'$t$', fontsize=11)
    ax.set_ylabel("Mean deviation scaled by stationary mean, " + r'$\frac{\left < x(t) - x(0) \right >_{T}}{\left < x \right >}$', fontsize=10)
    ax.set_title(plot_utils.sim_type_label_dict[sim_types[ax_idx]], fontsize=13)


fig.subplots_adjust(hspace=0.25, wspace=0.35)
fig_name = "%scompare_sojourn_sim.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()


#cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))
