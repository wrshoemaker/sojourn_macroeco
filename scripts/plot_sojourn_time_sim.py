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




sojourn_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_dict_path, "rb"))
data_type = 'linear'



fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

ax_demog = plt.subplot2grid((1,2), (0,0))
ax_slm = plt.subplot2grid((1,2), (0,1))


def make_list_from_count_dict(dict_):

    counts = []
    for key, value in dict_.items():
        counts.extend([key]*value)

    return counts



cv_all = []
mean_x_all = []
cv_x_all = []
for m in sojourn_dict['demog'].keys():

    for r in sojourn_dict['demog'][m].keys():

        if (1/r != float(100)):
            continue

        for D in sojourn_dict['demog'][m][r].keys():
                
            mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)
            mean_x_all.append(mean_x)
            cv_x_all.append(cv_x)


for sigma in sojourn_dict['slm'].keys():

    for k in sojourn_dict['slm'][sigma].keys():

        for tau in sojourn_dict['slm'][sigma][k].keys():
            
            if (tau != float(100)):
                continue

            mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)
            mean_x_all.append(mean_x)
            cv_x_all.append(cv_x)


cmap = cm.ScalarMappable(norm = colors.LogNorm(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))


for m in sojourn_dict['demog'].keys():

    for r in sojourn_dict['demog'][m].keys():
        
        # timescale
        if (1/r != float(100)):
            continue


        for D in sojourn_dict['demog'][m][r].keys():

            sojourn_dict_i = sojourn_dict['demog'][m][r][D]['linear']

            x = numpy.asarray(list(sojourn_dict_i.keys()))
            pdf_x = numpy.asarray(list(sojourn_dict_i.values()))
            pdf_x = pdf_x/sum(pdf_x)

            #sojourn_counts_i = make_list_from_count_dict(sojourn_dict_i)

            print('demo', m, D)

            mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m, r, D)

            #ax_slm.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)

            ax_demog.plot(x, pdf_x, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(cv_x))


            #ax_demog.bar(list(sojourn_dict_i.keys()), pdf)

            #ax_demog.hist(sojourn_counts_i, bins=30, density=True, histtype='step', lw=1, alpha=0.9, fill=False, color='dodgerblue')

            #ax_demog.scatter(x, pdf_x, s=20, alpha=0.7, color='dodgerblue')
            #ax_demog.plot(x, pdf_x, lw=0.5, ls='-', alpha=0.4, color='dodgerblue')



for sigma in sojourn_dict['slm'].keys():

    for k in sojourn_dict['slm'][sigma].keys():

        for tau in sojourn_dict['slm'][sigma][k].keys():

            if (tau != float(100)):
                continue

            sojourn_dict_i = sojourn_dict['slm'][sigma][k][tau]['linear']

            x = numpy.asarray(list(sojourn_dict_i.keys()))
            pdf_x = numpy.asarray(list(sojourn_dict_i.values()))
            pdf_x = pdf_x/sum(pdf_x)


            mean_x, cv_x = simulation_utils.calculate_mean_and_cv_slm(k,sigma)

            #ax_slm.plot(s_range, mean_run_deviation_target_rescaled_by_mean, lw=0.5, ls='-', alpha=0.3)

            ax_slm.plot(x, pdf_x, lw=0.5, ls='-', alpha=0.4, color=cmap.to_rgba(cv_x))


            #sojourn_counts_i = make_list_from_count_dict(sojourn_dict_i)

            #ax_slm.hist(sojourn_counts_i, bins=30, density=True, histtype='step', lw=1, alpha=0.9, fill=False, color='dodgerblue')



ax_demog.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
ax_demog.set_ylabel("Probability density", fontsize=11)

ax_slm.set_xlabel("Sojourn time, " + r'$T$', fontsize=11)
ax_slm.set_ylabel("Probability density", fontsize=11)

ax_demog.set_title("Demographic, " + r'$r^{-1} = $' + '100', fontsize=13)
ax_slm.set_title("SLM, " + r'$\tau = $' + '100', fontsize=13)

ax_demog.set_xlim([1, 1100])
ax_slm.set_xlim([1, 1100])


ax_demog.set_xscale('log', base=10)
ax_slm.set_xscale('log', base=10)

ax_demog.set_yscale('log', base=10)
ax_slm.set_yscale('log', base=10)



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_sim.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()

