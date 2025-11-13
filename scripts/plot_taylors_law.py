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

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))

        x_mean_all = []
        x_var_all = []
        for key, value in mle_dict[dataset][host].items():

            rel_abundance = numpy.asarray(value['rel_abundance'])

            x_mean = value['x_mean']
            x_var = (value['x_std'])**2

            ax.scatter(x_mean, x_var, color=plot_utils.host_color_dict[dataset][host], alpha=0.8, s=10)

            x_mean_all.append(x_mean)
            x_var_all.append(x_var) 
        

        if host_idx == 0:
            ax.set_ylabel('Variance of rel. abundance, ' + r'$\mathrm{Var}(x_{i})$', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Mean relative abundance, ' + r'$\bar{x}_{i}$', fontsize=12)


        ax.set_title(plot_utils.host_name_dict[dataset][host], fontsize=12)
        log10_x_mean_all = numpy.log10(x_mean_all)
        log10_x_var_all = numpy.log10(x_var_all)

        slope, intercept, r_value, p_value, std_err = linregress(log10_x_mean_all, log10_x_var_all)
        intercept_exponent_2 = numpy.mean(log10_x_var_all) - 2*numpy.mean(log10_x_mean_all)

        x_range_ =  numpy.linspace(min(log10_x_mean_all), max(log10_x_mean_all), 10000)
        y_fit_range = slope*x_range_ + intercept
        y_fit_range_exponent_2 = 2*x_range_ + intercept_exponent_2

        ax.plot(10**x_range_, 10**y_fit_range, ls='-', lw=2.5, c='k', label='Fitted exponent = %0.2f' % slope)
        ax.plot(10**x_range_, 10**y_fit_range_exponent_2, ls=':', lw=2.5, c='k', label='Exponent = 2')
        #ax.set_xlim([min(x_mean_all), max(x_mean_all)])

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)

        #if (host_idx == 0) and (dataset_idx==0):

        ax.legend(loc='upper left', fontsize=10)



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%staylors_law.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()