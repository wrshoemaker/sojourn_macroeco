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


#dist_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_fixed_moments_dict_path, "rb"))

#dist_dict = pickle.load(open('%ssojourn_time_dist_sim_fixed_moments_bdm_dict.pickle' % config.data_directory, "rb"))
dist_dict = pickle.load(open(simulation_utils.sojourn_time_dist_sim_fixed_moments_dict_path, "rb"))



fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

ax_bdm = plt.subplot2grid((1,2), (0,0))
ax_slm = plt.subplot2grid((1,2), (0,1))

target_mean = 10000
cv_all = list(dist_dict[target_mean].keys())
tau_all = list(dist_dict[target_mean][cv_all[0]].keys())
tau_all.sort()


cmap = cm.ScalarMappable(norm = colors.LogNorm(min(tau_all), max(tau_all)), cmap = plt.get_cmap('Blues'))


ax_bdm.set_title('BDM\n' + r'$\bar{x} = $' + str(target_mean), fontsize=12)
ax_slm.set_title('SLM\n' + r'$\bar{x} = $' + str(target_mean), fontsize=12)

ax_bdm.axhline(y=1000, ls=':', lw=2, c='k', label='Duration of random walk')
ax_slm.axhline(y=1000, ls=':', lw=2, c='k', label='Duration of random walk')

for tau in tau_all:

    print(tau)


    #tau_all = list(dist_dict[target_mean][cv].keys())
    #tau_all.sort()

    #print(dist_dict_bdm[target_mean][cv].keys())

    mean_sojourn_time_bdm = [dist_dict[target_mean][cv][tau]['bdm']['mean_sojourn_time'] for cv in cv_all]
    mean_sojourn_time_slm = [dist_dict[target_mean][cv][tau]['slm']['mean_sojourn_time'] for cv in cv_all]

    ax_bdm.scatter(cv_all, mean_sojourn_time_bdm, c=cmap.to_rgba(tau), s=10, alpha=1, label=r'$\tau = $' + str(round(tau, 3)))
    ax_bdm.plot(cv_all, mean_sojourn_time_bdm, c=cmap.to_rgba(tau), lw=2, ls='-')

    ax_slm.scatter(cv_all, mean_sojourn_time_slm, c=cmap.to_rgba(tau), s=10, alpha=1)
    ax_slm.plot(cv_all, mean_sojourn_time_slm, c=cmap.to_rgba(tau), lw=2, ls='-')

    #print(mean_sojourn_time_slm)


ax_bdm.set_xscale('log', base=10)
ax_slm.set_xscale('log', base=10)

ax_bdm.set_yscale('log', base=10)
ax_slm.set_yscale('log', base=10)


ax_bdm.legend(loc='upper left', fontsize=7)


ax_bdm.set_xlabel("CV", fontsize=11)
ax_slm.set_xlabel("CV", fontsize=11)

ax_bdm.set_ylabel("Mean sojourn time", fontsize=11)
ax_slm.set_ylabel("Mean sojourn time", fontsize=11)

ax_bdm.set_ylim([1, 1100])
ax_slm.set_ylim([1, 1100])




fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssojourn_time_dist_fixed_mean.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()
