import pickle
import sys
import numpy
import data_utils
import plot_utils
import simulation_utils

import config
from scipy import integrate, stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import plot_sojourn_dist_data_mix
import data_utils


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


max_sojourn_time = 50

#sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = plot_sojourn_dist_data_mix.make_null_gamma_sojourn_time_dist()


def calculate_mean_area(run_dict):

    run_sojourn_integral = []
    for run_length, run_sojourn in run_dict.items():

        for run_sojourn_j in run_sojourn:

            run_sojourn_j = numpy.asarray(run_sojourn_j)
            run_sojourn_integral_j = integrate.simpson(run_sojourn_j, numpy.linspace(0, 1, num=len(run_sojourn_j), endpoint=True))

            run_sojourn_integral.append(run_sojourn_integral_j)

    mean_run_sojourn_integral = numpy.mean(run_sojourn_integral)

    return mean_run_sojourn_integral


target_asv_dict = {}

for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    target_asv_dict[dataset] = {}

    for host_idx, host in enumerate(list(mle_dict[dataset].keys())):
        target_asv_dict[dataset][host] = []



#mean_area_all = []
asv_all = []
dataset_all_area = []
host_all_area = []
for dataset in data_utils.dataset_all:
    
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host in host_all:

        for asv in  mle_dict[dataset][host].keys():

            # total number of samples (i.e., maximum # draws)
            n_samples = len(mle_dict[dataset][host][asv]['rel_abundance'])
            
            # what we need for the null distribution (cares about # samples, not # days)
            run_lengths = mle_dict[dataset][host][asv]['run_lengths']
            days_run_lengths = mle_dict[dataset][host][asv]['days_run_lengths']

            

            if max(days_run_lengths) >= max_sojourn_time:

                #print(days_run_lengths)

                #print( )

                print(list(zip(days_run_lengths,mle_dict[dataset][host][asv]['days_run_starts'] )))

                asv_all.append(asv)
                dataset_all_area.append(dataset)
                host_all_area.append(host)

                #target_asv_dict[dataset][host].append(asv)

                # area under integral
                #run_dict = mle_dict[dataset][host][asv]['run_dict']
                #if run_dict is not None:
                #    mean_area = calculate_mean_area(run_dict)
                



#mean_area_all = numpy.asarray(mean_area_all)
asv_all = numpy.asarray(asv_all)
dataset_all_area = numpy.asarray(dataset_all_area)
host_all_area = numpy.asarray(host_all_area)

idx_nested = data_utils.chunk_list(list(range(len(asv_all))), 3)

#largest_area_all_idx = numpy.argpartition(mean_area_all, -10)[-10:]
#largest_area_all_idx = largest_area_all_idx[numpy.argsort(mean_area_all[largest_area_all_idx])[::-1]]
#asv_largest_area_all = asv_all[largest_area_all_idx]
#dataset_largest_area_all = dataset_all_area[largest_area_all_idx]
#host_largest_area_all = host_all_area[largest_area_all_idx]


n_rows = len(idx_nested)
n_cols = len(idx_nested[0])

fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
#fig.suptitle("Taylor's Law", fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down


for row_idx, row_list in enumerate(idx_nested):

    for col_idx, idx in enumerate(row_list):

        ax = plt.subplot2grid((n_rows, n_cols), (row_idx, col_idx))

        dataset = dataset_all_area[idx]
        host = host_all_area[idx]
        asv = asv_all[idx]

        for key, value in mle_dict[dataset][host].items():

            x_mean = value['x_mean']
            x_std = value['x_std']
            x_cv = x_std/x_mean

            rescaled_rel_abund = numpy.asarray(value['rel_abundance'])/x_mean
            log_rescaled_rel_abund = numpy.log(rescaled_rel_abund)

            mean_log_rescaled_rel_abund = simulation_utils.calculate_mean_log_rescaled_gamma(x_cv)
            deviation_log_rel_abund = log_rescaled_rel_abund - mean_log_rescaled_rel_abund

            if key == asv:
                lw=1.5
                alpha=1
                c='k'
                zorder = 2
            else:
                lw=0.5
                alpha=0.2
                c=plot_utils.host_color_dict[dataset][host]
                zorder=1


            ax.plot(value['days'], log_rescaled_rel_abund, ls='-', lw=lw, alpha=alpha, c=c, zorder=zorder)
            ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)

        if host_idx == 0:
            ax.set_ylabel(r'$y(t)  - \bar{y}$', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Time (days)', fontsize=12)

        ax.axhline(y=0, lw=2, ls=':', c='k')




fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stimeseries_outliers.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()



