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

from scipy.stats import loggamma, mode, spearmanr


from itertools import combinations  




mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


n_rows = len(data_utils.dataset_all)
n_cols = 4
fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
#fig.suptitle("Sojourn trajectories", fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down


corr_of_log = True


for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        asv_pairs = list(combinations(list(mle_dict[dataset][host].keys()), 2))

        delta_log_mean_all = []
        corr_all = []

        for asv_pair in asv_pairs:

            x_mean_1 = mle_dict[dataset][host][asv_pair[0]]['x_mean']
            x_mean_2 = mle_dict[dataset][host][asv_pair[1]]['x_mean']

            rel_abundance_1 = numpy.asarray(mle_dict[dataset][host][asv_pair[0]]['rel_abundance'])
            rel_abundance_2 = numpy.asarray(mle_dict[dataset][host][asv_pair[1]]['rel_abundance'])

            rel_abundance_1 = rel_abundance_1/x_mean_1
            rel_abundance_2 = rel_abundance_2/x_mean_2

            delta_log_mean_all.append(numpy.abs(numpy.log10(x_mean_1) - numpy.log10(x_mean_2)))

            #print(mle_dict[dataset][host][asv_pair[0]].keys())

            if corr_of_log == True:
                rel_abundance_1 = numpy.log10(rel_abundance_1)
                rel_abundance_2 = numpy.log10(rel_abundance_2)

            corr_all.append(numpy.corrcoef(rel_abundance_1, rel_abundance_2)[0,1])


        delta_log_mean_all = numpy.asarray(delta_log_mean_all)
        corr_all = numpy.asarray(corr_all)

        rho, p_value = spearmanr(delta_log_mean_all, corr_all)
        print(rho, p_value)

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))

        ax.scatter(delta_log_mean_all, corr_all, color=plot_utils.host_color_dict[dataset][host], alpha=0.6, s=8)

        ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)
        ax.set_ylim([0,3.7])
        ax.set_ylim([-1,1])

        ax.set_xlabel('Absolute difference in\nlog10 mean abundance', fontsize=12)
        ax.set_ylabel('Corr. of log10 abundance', fontsize=12)

        ax.text(0.03, 0.1, "Spearman's rho = " + str(round(rho, 4)), fontsize=10, transform=ax.transAxes)
        ax.text(0.03, 0.03, "P = " + str(round(p_value, 6)), fontsize=10, transform=ax.transAxes)


        nbins = 20
        bins = numpy.linspace(delta_log_mean_all.min(), delta_log_mean_all.max(), nbins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_idx = numpy.digitize(delta_log_mean_all, bins) - 1

        y_mean = numpy.array([corr_all[bin_idx == i].mean() if numpy.any(bin_idx == i) else numpy.nan for i in range(nbins)])

        ax.plot(bin_centers, y_mean, lw=2, ls='--', c='k', zorder=2)


fig.subplots_adjust(hspace=0.45, wspace=0.35)
fig_name = "%sdelta_mean_vs_corr.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()