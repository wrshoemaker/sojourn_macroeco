
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


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))
#min_n_autocorr_values = 6

n_rows = len(data_utils.dataset_all)
n_cols = 4
fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
        ax.set_title('%s, %s' % (dataset, host), fontsize=11)

        delay_days_all_all = []
        autocorr_all_all = []

        for key, value in mle_dict[dataset][host].items():
            
            x_mean = value['x_mean']
            rel_abundance = numpy.asarray(value['rel_abundance'])
            ln_rescaled_rel_abundance = numpy.log(rel_abundance/x_mean)
            days = numpy.asarray(value['days'])



            delay_days_all, autocorr_all = stats_utils.autocorrelation_by_days(ln_rescaled_rel_abundance, days, min_n_autocorr_values=5)

            delay_days_all = numpy.asarray(delay_days_all)
            autocorr_all = numpy.asarray(autocorr_all)

            if max(delay_days_all) < 3:
                continue
            
            ax.plot(delay_days_all, autocorr_all, lw=1, ls='-', c=plot_utils.host_color_dict[dataset][host], alpha=0.2)

            delay_days_all_all.extend(delay_days_all.tolist())
            autocorr_all_all.extend(autocorr_all.tolist())


        # remove frame if nothing was plotted 
        if len(delay_days_all_all) == 0:
            ax.set_axis_off()
            continue

        delay_days_all_all = numpy.asarray(delay_days_all_all)
        autocorr_all_all = numpy.asarray(autocorr_all_all)

        # plot example
        if len(delay_days_all_all) != 0:
            days_delay_range_ou = numpy.arange(0, max(delay_days_all_all)+1)
            #autocorr_ou = numpy.exp(-days_delay_range_ou/1)
            ax.plot(days_delay_range_ou, numpy.exp(-days_delay_range_ou/2), lw=2, ls='-', c='k', alpha=1, label='OU, ' + r'$\tau = 2$')
            ax.plot(days_delay_range_ou, numpy.exp(-days_delay_range_ou/1), lw=2, ls='--', c='k', alpha=1, label='OU, ' + r'$\tau = 1$')
            ax.plot(days_delay_range_ou, numpy.exp(-days_delay_range_ou/0.5), lw=2, ls='dashdot', c='k', alpha=1, label='OU, ' + r'$\tau = 0.5$')

            ax.set_ylim([min(autocorr_all_all), 1])
            ax.set_xlim([0, max(delay_days_all_all)])
            #ax.legend(loc='lower left')


            # get empirical mean
            #delay_days_set = list(set(delay_days_all_all))
            #mean_autocorr = [ numpy.mean(autocorr_all[delay_days_all==d]) for d in delay_days_set]
            #ax.plot(delay_days_set, mean_autocorr, lw=3, ls='--', c='k', alpha=1, zorder=3)

        
        
        

        ax.set_xlim([0,6])
        ax.set_ylim([-1,1])

        ax.axhline(y=0 , ls=':', lw=2, c='k', zorder=2, label=r'$\rho=0$')

        if host_idx + dataset_idx == 0:
            ax.legend(loc='lower left')


        if (host_idx == 0) and (len(delay_days_all_all)!=0):
            ax.set_ylabel("Autocorrelation, " + r'$\rho(\delta t)$', fontsize=10) 
            
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)) and (len(delay_days_all_all)!=0):
            ax.set_xlabel('Days between observations, ' + r'$\delta t$', fontsize=10)


        



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%sautocorr.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()

