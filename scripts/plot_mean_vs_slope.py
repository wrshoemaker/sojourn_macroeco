import pickle
import sys
import numpy
import data_utils
import plot_utils
import config

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import loggamma, mode, linregress



n_rows = len(data_utils.dataset_all)
n_cols = 4


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        x_mean_all = []
        slope_all = []
        for key, value in mle_dict[dataset][host].items():

            log_rel_abundance = numpy.log(value['rel_abundance'])
            days = numpy.asarray(value['days'])

            logfold_per_day = (log_rel_abundance[1:] - log_rel_abundance[:-1]) / (days[1:] - days[:-1])
            slope, intercept, r_value, p_value, std_err = linregress(log_rel_abundance[:-1], logfold_per_day)

            x_mean_all.append(value['x_mean'])
            slope_all.append(slope)
            #sprint(rel_abundance)


        slope, intercept, r_value, p_value, std_err = linregress(numpy.log10(x_mean_all), slope_all)
        print(p_value)

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
        ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)
        ax.scatter(x_mean_all, slope_all, color=plot_utils.host_color_dict[dataset][host], alpha=0.8, s=10)

        ax.set_xscale('log', base=10)


        if host_idx == 0:
            ax.set_ylabel('Slope of ' + r'$ \mathrm{ln} \, x_{i}(t)$' + ' vs. ' + r'$\frac{1}{x_{i}} \frac{\Delta x_{i}}{\Delta t}$', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Mean relative abundance, ' + r'$\bar{x}_{i}$', fontsize=12)



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%smean_vs_slope.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()