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



sojourn_time=15


fig = plt.figure(figsize = (8, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

ax_bdm = plt.subplot2grid((1,2), (0,0))
ax_slm = plt.subplot2grid((1,2), (0,1))


mean = list(dist_dict.keys())[0]
cv_all = list(dist_dict[mean].keys())
tau_all = list(dist_dict[mean][cv_all[0]].keys())
tau_all.sort()

cmap = cm.ScalarMappable(norm = colors.LogNorm(min(cv_all), max(cv_all)), cmap = plt.get_cmap('Blues'))

for cv in cv_all:

    norm_constant_bdm_all = []
    norm_constant_slm_all = []

    for tau in tau_all:

        run_length_bdm = dist_dict[mean][cv][tau]['bdm']['run_length']
        run_length_slm = dist_dict[mean][cv][tau]['slm']['run_length']

        run_length_bdm = numpy.asarray(run_length_bdm)
        run_length_slm = numpy.asarray(run_length_slm)

        norm_constant_bdm = dist_dict[mean][cv][tau]['bdm']['norm_constant']
        norm_constant_slm = dist_dict[mean][cv][tau]['slm']['norm_constant']

        sojourn_time_bdm_idx = numpy.where(run_length_bdm == sojourn_time)[0][0]
        sojourn_time_slm_idx = numpy.where(run_length_slm == sojourn_time)[0][0]

        norm_constant_bdm_all.append(norm_constant_bdm[sojourn_time_bdm_idx])
        norm_constant_slm_all.append(norm_constant_slm[sojourn_time_slm_idx])

    

    ax_bdm.plot(tau_all, norm_constant_bdm_all, lw=2, ls='-', c=cmap.to_rgba(cv), alpha=1)
    ax_slm.plot(tau_all, norm_constant_slm_all, lw=2, ls='-', c=cmap.to_rgba(cv), alpha=1)


sim_types = ['bdm', 'slm']

for ax_idx, ax in enumerate([ax_bdm, ax_slm]):

    ax.set_ylim([10,800000])

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel(r'$\tau$', fontsize=11)
    ax.set_ylabel("Integral over sojourn period", fontsize=10)
    ax.set_title(plot_utils.sim_type_label_dict[sim_types[ax_idx]], fontsize=13)


fig.subplots_adjust(hspace=0.25, wspace=0.35)
fig_name = "%stau_vs_sojourn_integral_sim.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()

