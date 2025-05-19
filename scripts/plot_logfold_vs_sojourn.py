import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
from collections import Counter
import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma
from scipy import integrate
from scipy.optimize import minimize_scalar


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
import theory_utils


environment = 'gut'


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

host_count = 0

fig = plt.figure(figsize = (12, 4)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    mean_logfold_all = []
    mean_sojourn_time_all = []
    for host in host_all:

        for key, value in mle_dict[dataset][host].items():

            rel_abundance = numpy.asarray(value['rel_abundance'])
            rescaled_log_rel_abundance = numpy.log(rel_abundance/value['x_mean'])

            days = numpy.asarray(value['days'])
            mean_logfold = numpy.mean((rescaled_log_rel_abundance[1:] - rescaled_log_rel_abundance[:-1])/(days[1:] - days[:-1]))

            mean_sojourn_time = numpy.mean(value['days_run_lengths'])

            mean_logfold_all.append(mean_logfold)
            mean_sojourn_time_all.append(mean_sojourn_time)


    mean_logfold_all = numpy.abs(mean_logfold_all)

    ax = plt.subplot2grid((1, len(data_utils.dataset_all)), (0, dataset_idx))
    ax.scatter(mean_logfold_all, mean_sojourn_time_all, lw=0.5, ls='-', s=10, alpha=0.8, color=plot_utils.dataset_color_dict[dataset])
    ax.set_yscale('log', base=10)

    log10_mean_sojourn_time_all = numpy.log10(mean_sojourn_time_all)
    slope, intercept, r_value, p_value, std_err = data_utils.stats.linregress(mean_logfold_all, log10_mean_sojourn_time_all)

    print(slope, p_value)


    #mean_logfold_all_all.extend(mean_logfold_all)
    #mean_sojourn_time_all_all.extend(mean_sojourn_time_all)



#ax.scatter(mean_logfold_all, mean_sojourn_time_all)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%smean_logfold_vs_mean_sojourn.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()
    