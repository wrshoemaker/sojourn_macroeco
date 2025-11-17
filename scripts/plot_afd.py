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

from scipy.stats import loggamma, mode


n_rows = len(data_utils.dataset_all)
n_cols = 4


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

fig.suptitle("Abundance Fluctuation Distribution (AFD)", fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))

        rescaled_log_rel_abundance_all = []
        hist_all = []

        for key, value in mle_dict[dataset][host].items():

            rel_abundance = numpy.asarray(value['rel_abundance'])

            log_rel_abundance = numpy.log(rel_abundance)
            rescaled_log_rel_abundance = (log_rel_abundance - numpy.mean(log_rel_abundance))/numpy.std(log_rel_abundance)
            rescaled_log_rel_abundance_all.append(rescaled_log_rel_abundance)

            hist, bins = data_utils.get_hist_and_bins(rescaled_log_rel_abundance_all, n_bins=10)

            ax.plot(bins, hist, ls='-', lw=0.5, alpha=0.2, c=plot_utils.host_color_dict[dataset][host])
            #ax.set_title(plot_utils.host_name_dict[dataset][host], fontsize=12)
            ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)

            hist_all.append(hist)
  

        if host_idx == 0:
            ax.set_ylabel('Probability density', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Rescaled log-transformed rel. abund.', fontsize=12)


        # fit gamma
        rescaled_log_rel_abundance_all = numpy.concatenate(rescaled_log_rel_abundance_all).ravel()
        hist_all = numpy.concatenate(hist_all).ravel()

        loggamma_fit = loggamma.fit(rescaled_log_rel_abundance_all)
        #gammalog = k*k_trigamma*x_range - np.exp(np.sqrt(k_trigamma)*x_range + k_digamma) - np.log(special.gamma(k)) + k*k_digamma + np.log10(np.exp(1))
        x_range = numpy.linspace(min(rescaled_log_rel_abundance_all), max(rescaled_log_rel_abundance_all), 10000)

        ax.plot(x_range, loggamma(loggamma_fit[0], loggamma_fit[1], loggamma_fit[2]).pdf(x_range), 'k', label='Gamma', lw=2)

        ax.set_xlim([min(rescaled_log_rel_abundance_all), max(rescaled_log_rel_abundance_all)])
        ax.set_ylim([min(hist_all), max([max(hist_all), max(loggamma(loggamma_fit[0], loggamma_fit[1], loggamma_fit[2]).pdf(x_range))])])
        ax.set_yscale('log', base=10)


        if (host_idx == 0) and (dataset_idx==0):

            ax.legend(loc='upper left', fontsize=10)





fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%safd.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()