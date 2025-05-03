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

import simulation_utils


import theory_utils 


#min_x = 10
#max_x = 1000
#lambda_ = 10

k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(50, 2)


fig, ax = plt.subplots(figsize=(6,4))

max_t=800

for tau in [0.1, 1, 6]:

    t_range_slm, prob_t_slm = theory_utils.predict_sojourn_dist_slm(100000, k, sigma, tau, 10000)
    t_range_bdm, prob_t_bdm = theory_utils.predict_sojourn_dist_bdm(100000, m, phi, tau, 10000)

    t_range_slm = t_range_slm[:max_t]
    prob_t_slm = prob_t_slm[:max_t]
    prob_t_slm = prob_t_slm/sum(prob_t_slm)

    t_range_bdm = t_range_bdm[:max_t]
    prob_t_bdm = prob_t_bdm[:max_t]
    prob_t_bdm = prob_t_bdm/sum(prob_t_bdm)
    


    ax.plot(t_range_bdm, prob_t_bdm, c='k', lw=2.5, linestyle='-', zorder=2)



#ax.scatter(run_lengths, run_sum_all, s=10, alpha=0.4, c='k', zorder=1)
ax.set_ylim([0.00001, 1])



ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.set_xlabel('Sojourn time (days), ' + r'$T$', fontsize=10)
ax.set_ylabel("Probability density (not properly rescaled)", fontsize=10)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stheory_test.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()