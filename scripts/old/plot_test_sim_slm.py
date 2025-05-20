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


numpy.random.seed(123456789)


tau = 10

n_days=1000

k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(1000, 0.5)

time = numpy.arange(1, n_days+1)
x_matrix_slm = simulation_utils.simulate_slm_trajectory(n_days=n_days, n_reps=10, k=k, sigma=sigma, tau=tau)


fig, ax = plt.subplots(figsize=(6,4))

ax.plot(time, x_matrix_slm[1:,0], lw=1, ls='-', alpha=0.6, c='k')
ax.plot(time, x_matrix_slm[1:,1], lw=1, ls='-', alpha=0.6, c='k')
ax.plot(time, x_matrix_slm[1:,2], lw=1, ls='-', alpha=0.6, c='k')
ax.plot(time, x_matrix_slm[1:,3], lw=1, ls='-', alpha=0.6, c='k')
ax.plot(time, x_matrix_slm[1:,4], lw=1, ls='-', alpha=0.6, c='k')


ax.set_yscale('log', base=10)

fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stest_sim_slm.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()


