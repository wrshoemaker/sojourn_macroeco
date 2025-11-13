from __future__ import division
import config
import os
import sys
import subprocess
import random
import numpy
import re
import gzip
from collections import Counter
import itertools
import scipy.stats as stats
from scipy.special import digamma, gamma, erf, loggamma, hyperu, polygamma

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import integrate
import data_utils
import plot_utils
import stats_utils
import pickle


# load dictionaries
mle_null_dict_path = '%smle_null_dict.pickle' % config.data_directory
res_ret_dict_path = '%sres_ret_dict.pickle' % config.data_directory

mle_null_dict = pickle.load(open(mle_null_dict_path, "rb"))
res_ret_dict = pickle.load(open(res_ret_dict_path, "rb"))


colors_dict = {'sojourn':'#87CEEB', 'residence': '#FFA500', 'return':'#FF6347'}


legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_dict['residence'], label=r'$t_{\mathrm{res}}$', markersize=15),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_dict['return'], label=r'$t_{\mathrm{ret}}$', markersize=15),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_dict['sojourn'], label=r'$\mathcal{T}$', markersize=15)]



fig, ax = plt.subplots(figsize=(4,8))

measure_all = ['sojourn', 'residence', 'return']




y_ax_count = 0
y_tick_labels = []

x_y_all_dict = {}
for measure in measure_all:
    x_y_all_dict[measure] = {}
    x_y_all_dict[measure]['x'] = []
    x_y_all_dict[measure]['y'] = []

for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    
    read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, 'gut')
    host_all = list(set(host_status))
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        for measure in measure_all:

            if measure == 'sojourn':
                x_range = mle_null_dict[dataset][host]['prob_sojourn']['x_range_pdf']
                pdf = mle_null_dict[dataset][host]['prob_sojourn']['days_pdf']
                x_range_null = mle_null_dict[dataset][host]['prob_sojourn']['x_range_pdf_null']
                pdf_null = mle_null_dict[dataset][host]['prob_sojourn']['days_pdf_null']


            elif measure == 'residence':
                x_range = res_ret_dict[dataset][host][True]['x_range_pdf']
                pdf = res_ret_dict[dataset][host][True]['days_pdf']
                x_range_null = res_ret_dict[dataset][host][True]['x_range_pdf_null']
                pdf_null = res_ret_dict[dataset][host][True]['days_pdf_null']

            else:
                x_range = res_ret_dict[dataset][host][False]['x_range_pdf']
                pdf = res_ret_dict[dataset][host][False]['days_pdf']
                x_range_null = res_ret_dict[dataset][host][False]['x_range_pdf_null']
                pdf_null = res_ret_dict[dataset][host][False]['days_pdf_null']

            x_range = numpy.asarray(x_range)
            pdf = numpy.asarray(pdf)
            x_range_null = numpy.asarray(x_range_null)
            pdf_null = numpy.asarray(pdf_null)

            common_ids = numpy.intersect1d(x_range, x_range_null)
            mask = numpy.isin(x_range, common_ids)
            mask_null = numpy.isin(x_range_null, common_ids)
            pdf_shared = pdf[mask]
            range_null_shared = pdf_null[mask_null]
            js_div = stats_utils.js_div(pdf_shared, range_null_shared)
            ax.scatter(js_div, y_ax_count,  color=colors_dict[measure], alpha=1, s=70, zorder=2)

            x_y_all_dict[measure]['x'].append(js_div)
            x_y_all_dict[measure]['y'].append(y_ax_count)


        y_ax_count += 1
        y_tick_labels.append(host)



for measure in measure_all:
    ax.plot(x_y_all_dict[measure]['x'], x_y_all_dict[measure]['y'],  color=colors_dict[measure], lw=2, zorder=1)



ax.set_xlabel("Jensen–Shannon divergence\nb/w observed and time-permuted PDFs", fontsize=12)
ax.set_yticks(list(range(len(y_tick_labels))))
ax.set_yticklabels(y_tick_labels, fontsize=10)
#ax.legend(handles=legend_elements, loc='center')


ax.legend(
    handles=legend_elements,
    loc='upper center',           # position relative to Axes
    bbox_to_anchor=(0.5, 1.07),   # x, y offset from Axes
    ncol=3,                       # number of legend columns
    fontsize=13,
    frameon=False                 # optional: remove box
)

fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%scompare_measures_jsd.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()