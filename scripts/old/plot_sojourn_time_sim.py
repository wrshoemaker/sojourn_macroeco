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


rescale_sojourn_time = False

if rescale_sojourn_time == True:
    rescale_sojourn_time_label = '_rescaled'
else:
    rescale_sojourn_time_label = ''

sojourn_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_dict_path, "rb"))
data_type = 'linear'


def get_stat_list_all(stat):

    stat_list_all = []

    for m in sojourn_dict['demog'].keys():

        for r in sojourn_dict['demog'][m].keys():

            for D in sojourn_dict['demog'][m][r].keys():
                    
                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)

                if stat == 'mean':
                    stat_list_all.append(mean_x)

                elif stat == 'cv':
                    stat_list_all.append(cv_x)

                else:
                    stat_list_all.append(1/r)


    for sigma in sojourn_dict['slm'].keys():

        for k in sojourn_dict['slm'][sigma].keys():

            for tau in sojourn_dict['slm'][sigma][k].keys():
                
                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)
                if stat == 'mean':
                    stat_list_all.append(mean_x)

                elif stat == 'cv':
                    stat_list_all.append(cv_x)

                else:
                    stat_list_all.append(tau)


    return stat_list_all



def plot_sojourn_dist_old():

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_demog = plt.subplot2grid((1,2), (0,0))
    ax_slm = plt.subplot2grid((1,2), (0,1))

    timescale_all = get_stat_list_all('timescale')
    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(timescale_all), max(timescale_all)), cmap = plt.get_cmap('Blues'))

    for m in sojourn_dict['demog'].keys():

        for r in sojourn_dict['demog'][m].keys():
            
            for D in sojourn_dict['demog'][m][r].keys():

                sojourn_dict_i = sojourn_dict['demog'][m][r][D]['linear']

                x = numpy.asarray(list(sojourn_dict_i.keys()))

                pdf_x = numpy.asarray(list(sojourn_dict_i.values()))
                pdf_x = pdf_x/sum(pdf_x)

                ax_demog.plot(x, pdf_x, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(1/r))

    for sigma in sojourn_dict['slm'].keys():

        for k in sojourn_dict['slm'][sigma].keys():

            for tau in sojourn_dict['slm'][sigma][k].keys():

                sojourn_dict_i = sojourn_dict['slm'][sigma][k][tau]['linear']

                x = numpy.asarray(list(sojourn_dict_i.keys()))

                pdf_x = numpy.asarray(list(sojourn_dict_i.values()))
                pdf_x = pdf_x/sum(pdf_x)

                ax_slm.plot(x, pdf_x, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(tau))


    ax_demog.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    ax_demog.set_ylabel("Probability density", fontsize=11)

    ax_slm.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    ax_slm.set_ylabel("Probability density", fontsize=11)

    ax_demog.set_title("Demographic", fontsize=13)
    ax_slm.set_title("SLM", fontsize=13)

    ax_demog.set_xlim([1, 1100])
    ax_slm.set_xlim([1, 1100])


    ax_demog.set_xscale('log', base=10)
    ax_slm.set_xscale('log', base=10)

    ax_demog.set_yscale('log', base=10)
    ax_slm.set_yscale('log', base=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_sim%s.png" % (config.analysis_directory, rescale_sojourn_time_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_sojourn_dist():

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    sojourn_dict = pickle.load(open('%ssojourn_time_dist_sim_fixed_moments_dict.pickle' % config.data_directory, 'rb'))

    mean = list(sojourn_dict.keys())[0]
    cv_all = list(sojourn_dict[mean].keys())

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

            #print(sojourn_dict[mean][cv][tau]['bdm']['slope'])

            x_bdm = numpy.asarray(list(sojourn_dict_bdm_i.keys()))
            x_slm = numpy.asarray(list(sojourn_dict_slm_i.keys()))

            #print(max(x_bdm), max(x_slm))


            if sum(x_bdm > 200) > 0:
                continue

            #if 1 not in x_bdm:
            #    continue

            pdf_x_bdm = numpy.asarray(list(sojourn_dict_bdm_i.values()))
            pdf_x_bdm = pdf_x_bdm/sum(pdf_x_bdm)

            x_bdm_sort_idx = numpy.argsort(x_bdm)
            x_bdm = x_bdm[x_bdm_sort_idx]
            pdf_x_bdm = pdf_x_bdm[x_bdm_sort_idx]


            ax_demog.plot(x_bdm, pdf_x_bdm, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(cv))


            if sum(x_slm > 200) > 0:
                continue

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

    #ax_demog.set_xlim([1, 1100])
    #ax_slm.set_xlim([1, 1100])

    ax_demog.set_xlim([1, 220])
    ax_slm.set_xlim([1, 220])


    ax_demog.set_xscale('log', base=10)
    ax_slm.set_xscale('log', base=10)

    ax_demog.set_yscale('log', base=10)
    ax_slm.set_yscale('log', base=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_sim%s.png" % (config.analysis_directory, rescale_sojourn_time_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_stat_vs_mean_sojourn_old():

    fig = plt.figure(figsize = (8, 12)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_demog_mean = plt.subplot2grid((3,2), (0,0))
    ax_demog_cv = plt.subplot2grid((3,2), (1,0))
    ax_demog_timescale = plt.subplot2grid((3,2), (2,0))

    ax_slm_mean = plt.subplot2grid((3,2), (0,1))
    ax_slm_cv = plt.subplot2grid((3,2), (1,1))
    ax_slm_timescale = plt.subplot2grid((3,2), (2,1))

    timescale_all = get_stat_list_all('timescale')

    ax_demog_mean.set_title('Demographic model', fontsize=12)
    ax_slm_mean.set_title('SLM', fontsize=12)

    for m in sojourn_dict['demog'].keys():

        for r in sojourn_dict['demog'][m].keys():
            
            for D in sojourn_dict['demog'][m][r].keys():

                sojourn_dict_i = sojourn_dict['demog'][m][r][D]['linear']

                mean_sojourn = data_utils.calculate_mean_from_count_dict(sojourn_dict_i)
                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)

                ax_demog_mean.scatter(mean_x, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')
                ax_demog_cv.scatter(cv_x, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')
                ax_demog_timescale.scatter(1/r, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')


    for sigma in sojourn_dict['slm'].keys():

        for k in sojourn_dict['slm'][sigma].keys():

            for tau in sojourn_dict['slm'][sigma][k].keys():

                sojourn_dict_i = sojourn_dict['slm'][sigma][k][tau]['linear']

                mean_sojourn = data_utils.calculate_mean_from_count_dict(sojourn_dict_i)
                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)

                ax_slm_mean.scatter(mean_x, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')
                ax_slm_cv.scatter(cv_x, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')
                ax_slm_timescale.scatter(tau, mean_sojourn, lw=0.5, ls='-', s=10, alpha=0.8, color='dodgerblue')


    #ax_demog.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    #ax_demog.set_ylabel("Probability density", fontsize=11)

    #ax_slm.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
    #ax_slm.set_ylabel("Probability density", fontsize=11)

    #ax_demog.set_title("Demographic", fontsize=13)
    #ax_slm.set_title("SLM", fontsize=13)

    #ax_demog.set_xlim([1, 1100])
    #ax_slm.set_xlim([1, 1100])

    ax_demog_mean.set_xlabel("Mean abundance", fontsize=11)
    ax_slm_mean.set_xlabel("Mean abundance", fontsize=11)

    ax_demog_cv.set_xlabel("CV of abundance", fontsize=11)
    ax_slm_cv.set_xlabel("CV of abundance", fontsize=11)

    ax_demog_timescale.set_xlabel("Growth timescale", fontsize=11)
    ax_slm_timescale.set_xlabel("Growth timescale", fontsize=11)


    ax_demog_mean.set_ylabel("Mean sojourn time", fontsize=11)
    ax_slm_mean.set_ylabel("Mean sojourn time", fontsize=11)
    ax_demog_cv.set_ylabel("Mean sojourn time", fontsize=11)
    ax_slm_cv.set_ylabel("Mean sojourn time", fontsize=11)
    ax_demog_timescale.set_ylabel("Mean sojourn time", fontsize=11)
    ax_slm_timescale.set_ylabel("Mean sojourn time", fontsize=11)



    ax_demog_mean.set_xscale('log', base=10)
    ax_demog_cv.set_xscale('log', base=10)
    ax_demog_timescale.set_xscale('log', base=10)

    ax_slm_mean.set_xscale('log', base=10)
    ax_slm_cv.set_xscale('log', base=10)
    ax_slm_timescale.set_xscale('log', base=10)

    ax_demog_mean.set_yscale('log', base=10)
    ax_demog_cv.set_yscale('log', base=10)
    ax_demog_timescale.set_yscale('log', base=10)

    ax_slm_mean.set_yscale('log', base=10)
    ax_slm_cv.set_yscale('log', base=10)
    ax_slm_timescale.set_yscale('log', base=10)

    #ax_demog.set_yscale('log', base=10)
    #ax_slm.set_yscale('log', base=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sstat_vs_mean_sojourn.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()







if __name__ == "__main__":


    print("Running...")

    #plot_sojourn_dist()
    #plot_stat_vs_mean_sojourn()

    plot_sojourn_dist()
