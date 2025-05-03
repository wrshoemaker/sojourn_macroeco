import os
import random
import copy
import config
import sys
import numpy
import math
import random
import pickle
from collections import Counter

import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma, erf
from scipy import interpolate

import stats_utils
import data_utils
import simulation_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch



sojourn_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_fixed_moments_analytic_dict_path, "rb"))


mean = list(sojourn_dict.keys())[0]
cv_all = list(sojourn_dict[mean].keys())
tau_all = list(sojourn_dict[mean][cv_all[0]].keys())

cmap_cv = cm.ScalarMappable(norm = colors.LogNorm(min(cv_all), max(cv_all)), cmap = plt.get_cmap('Blues'))

print(sojourn_dict[mean][cv_all[0]][tau_all[0]]['bdm'].keys())



def plot_sojourn_dist():

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_demog = plt.subplot2grid((1,2), (0,0))
    ax_slm = plt.subplot2grid((1,2), (0,1))

    tau_all = list(sojourn_dict[mean][cv_all[0]].keys())
    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(cv_all), max(cv_all)), cmap = plt.get_cmap('Blues'))

    for cv in cv_all:

        #if cv > 2:
        #    continue

        for tau in tau_all:

            #if tau != 0.1:
            #    continue

            sojourn_dict_bdm_i = sojourn_dict[mean][cv][tau]['bdm']['sojourn_time_count_dict']
            sojourn_dict_slm_i = sojourn_dict[mean][cv][tau]['slm']['sojourn_time_count_dict']

            print('bdm', cv, tau, sojourn_dict[mean][cv][tau]['bdm']['slope'])
            print('slm', cv, tau, sojourn_dict[mean][cv][tau]['slm']['slope'])

            x_bdm = numpy.asarray(list(sojourn_dict_bdm_i.keys()))
            x_slm = numpy.asarray(list(sojourn_dict_slm_i.keys()))

            #print(max(x_bdm), max(x_slm))


            #if sum(x_bdm > 200) > 0:
            #    continue


            pdf_x_bdm = numpy.asarray(list(sojourn_dict_bdm_i.values()))
            pdf_x_bdm = pdf_x_bdm/sum(pdf_x_bdm)

            x_bdm_sort_idx = numpy.argsort(x_bdm)
            x_bdm = x_bdm[x_bdm_sort_idx]
            pdf_x_bdm = pdf_x_bdm[x_bdm_sort_idx]


            ax_demog.plot(x_bdm, pdf_x_bdm, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(cv))


            #if sum(x_slm > 200) > 0:
            #    continue

            #if 1 not in x_slm:
            #    continue

            pdf_x_slm = numpy.asarray(list(sojourn_dict_slm_i.values()))
            pdf_x_slm = pdf_x_slm/sum(pdf_x_slm)

            x_slm_sort_idx = numpy.argsort(x_slm)
            x_slm = x_slm[x_slm_sort_idx]
            pdf_x_slm = pdf_x_slm[x_slm_sort_idx]

            ax_slm.plot(x_slm, pdf_x_slm, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(cv))



    ax_demog.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    ax_demog.set_ylabel("Probability density", fontsize=11)

    ax_slm.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    ax_slm.set_ylabel("Probability density", fontsize=11)

    ax_demog.set_title("BDM", fontsize=13)
    ax_slm.set_title("SLM", fontsize=13)

    #ax_demog.set_xlim([1, 220])
    #ax_slm.set_xlim([1, 220])

    ax_demog.set_xscale('log', base=10)
    ax_slm.set_xscale('log', base=10)

    ax_demog.set_yscale('log', base=10)
    ax_slm.set_yscale('log', base=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_analytic_sim.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_stat_vs_mean_sojourn():

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_bdm = plt.subplot2grid((1,2), (0,0))
    ax_slm = plt.subplot2grid((1,2), (0,1))

    ax_bdm.set_title('BDM', fontsize=12)
    ax_slm.set_title('SLM', fontsize=12)

    for cv in cv_all:

        mean_sojourn_bdm_all = []
        mean_sojourn_slm_all = []

        for tau in tau_all:
                
            sojourn_dict_bdm_i = sojourn_dict[mean][cv][tau]['bdm']['sojourn_time_count_dict']
            sojourn_dict_slm_i = sojourn_dict[mean][cv][tau]['slm']['sojourn_time_count_dict']

            mean_sojourn_bdm_i = data_utils.calculate_mean_from_count_dict(sojourn_dict_bdm_i)
            mean_sojourn_slm_i = data_utils.calculate_mean_from_count_dict(sojourn_dict_slm_i)

            mean_sojourn_bdm_all.append(mean_sojourn_bdm_i)
            mean_sojourn_slm_all.append(mean_sojourn_slm_i)


        ax_bdm.plot(tau_all, mean_sojourn_bdm_all, lw=1, ls='-', alpha=0.9, color=cmap_cv.to_rgba(cv))
        ax_slm.plot(tau_all, mean_sojourn_slm_all, lw=1, ls='-', alpha=0.9, color=cmap_cv.to_rgba(cv))

        print(mean_sojourn_slm_all)


    #ax_demog.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    #ax_demog.set_ylabel("Probability density", fontsize=11)

    #ax_slm.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    #ax_slm.set_ylabel("Probability density", fontsize=11)

    #ax_demog.set_title("Demographic", fontsize=13)
    #ax_slm.set_title("SLM", fontsize=13)

    #ax_demog.set_xlim([1, 1100])
    #ax_slm.set_xlim([1, 1100])


    ax_bdm.set_xlabel("Growth timescale", fontsize=11)
    ax_slm.set_xlabel("Growth timescale", fontsize=11)

    ax_bdm.set_ylabel("Mean sojourn time", fontsize=11)
    ax_slm.set_ylabel("Mean sojourn time", fontsize=11)

    ax_bdm.set_xscale('log', base=10)
    ax_slm.set_xscale('log', base=10)

    #ax_bdm.set_yscale('log', base=10)
    #ax_slm.set_yscale('log', base=10)

    #ax_demog.set_yscale('log', base=10)
    #ax_slm.set_yscale('log', base=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sstat_vs_mean_sojourn_analytic.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





if __name__ == "__main__":


    print("Running...")

    #plot_sojourn_dist()
    plot_stat_vs_mean_sojourn()