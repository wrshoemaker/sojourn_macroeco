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


min_x = 10
max_x = 1000
lambda_ = 10


fig, ax = plt.subplots(figsize=(6,4))


x_range =  numpy.logspace(numpy.log10(min_x) , numpy.log10(max_x), 10000)
#y_log10_fit_range = (slope*x_log10_range + intercept)

y = numpy.exp(-1*x_range*lambda_) / ((1 - numpy.exp(-2*x_range*lambda_))**(3/2))

ax.plot(x_range, y, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
#ax.scatter(run_lengths, run_sum_all, s=10, alpha=0.4, c='k', zorder=1)




ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

ax.set_xlabel('Sojourn time (days), ' + r'$T$', fontsize=10)
ax.set_ylabel("Probability density (not properly rescaled)", fontsize=10)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stheory_test.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()