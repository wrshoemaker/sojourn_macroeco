import pickle
import sys
import numpy
import data_utils
import plot_utils
import simulation_utils
import config

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import loggamma, mode





mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


n_rows = len(data_utils.dataset_all)
n_cols = 4
fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))

        for key, value in mle_dict[dataset][host].items():


            x_mean = value['x_mean']
            x_std = value['x_std']
            x_cv = x_std/x_mean

            rescaled_rel_abund = numpy.asarray(value['rel_abundance'])/x_mean
            log_rescaled_rel_abund = numpy.log(rescaled_rel_abund)

            #k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
            mean_log_rescaled_rel_abund = simulation_utils.calculate_mean_log_rescaled_gamma(x_cv)
            deviation_log_rel_abund = log_rescaled_rel_abund - mean_log_rescaled_rel_abund


            ax.plot(value['days'], log_rescaled_rel_abund, ls='-', lw=0.5, alpha=0.2, c=plot_utils.host_color_dict[dataset][host])
            ax.set_title(plot_utils.host_name_dict[dataset][host], fontsize=12)

  

        if host_idx == 0:
            ax.set_ylabel(r'$\mathrm{ln}\, \tilde{x}_{i}(t)  - \left < \mathrm{ln}\, \tilde{x}_{i} \right >$', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Time (days)', fontsize=12)


        ax.axhline(y=0, lw=2, ls=':', c='k')

        # fit gamma

        #ax.set_xlim([min(rescaled_log_rel_abundance_all), max(rescaled_log_rel_abundance_all)])
        #ax.set_ylim([min(hist_all), max([max(hist_all), max(loggamma(loggamma_fit[0], loggamma_fit[1], loggamma_fit[2]).pdf(x_range))])])
        #ax.set_yscale('log', base=10)


        #if (host_idx == 0) and (dataset_idx==0):
        #    ax.legend(loc='upper left', fontsize=10)





fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stimeseries.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()