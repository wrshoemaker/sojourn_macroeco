import pickle
import sys
import numpy
import data_utils
import stats_utils
import plot_utils
import config

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy import stats



max_delta_t = 20


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

n_rows = len(data_utils.dataset_all)
n_cols = 4
fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
#fig.suptitle('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down

for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        asv_all = list(mle_dict[dataset][host].keys())

        days = numpy.asarray(mle_dict[dataset][host][asv_all[0]]['days'])

        delta_t = days[1:] - days[:-1]

        x_range, pdf = stats_utils.get_pdf_from_counts(delta_t)


        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
        ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)
        
        ax.scatter(x_range, pdf, c=plot_utils.host_color_dict[dataset][host], lw=3, linestyle='-', zorder=2)



        print(max(x_range))
        ax.set_xlim([0, 60])
        #ax.set_ylim([0, 1])
        ax.axvspan(20, 60, color='gray', alpha=0.5, label='Excluded from analysis')

        ax.set_yscale('log', base=10)

        if host_idx == 0:
            #ax.set_ylabel('Fraction of sojourn trajectories ' + r'$\geq \mathcal{T}$', fontsize=9) 
            ax.set_ylabel("Probability density", fontsize=10)        
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Sampling interval (days), ' + r'$\delta t$', fontsize=10)

        if dataset_idx + host_idx == 0:
            ax.legend(loc = 'upper right')



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%ssampling_time_dist.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()