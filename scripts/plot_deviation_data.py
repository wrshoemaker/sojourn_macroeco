
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

import stats_utils
import simulation_utils


environment = 'gut'
occupancy_min = 1.0


ax_idx_all = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1), (4,0), (4,1)]

slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sim_dict_path = '%sfluctuation_sim_dict.pickle' %config.data_directory

mle_dict_path = '%smle_dict.pickle' %config.data_directory




def make_mle_dict():

    mle_dict = {}

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        mle_dict[dataset] = {}

        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, environment)

        host_all = list(set(host_status))
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            mle_dict[dataset][host] = {}

            # function subsets ASVs that are actually present
            read_counts_host, days_host, asv_names_host = data_utils.subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)
            rel_read_counts_host = (read_counts_host/read_counts_host.sum(axis=0))

            #occupancy_min_idx = numpy.sum(rel_read_counts_host>0, axis=1)/len(days_host) >= occupancy_min   

            #rel_read_counts_host_subset = rel_read_counts_host[occupancy_min_idx,:]
            #asv_names_host_subset = asv_names_host[occupancy_min_idx]

            total_abundance = numpy.sum(read_counts_host, axis=0)

            for asv_names_host_subset_i_idx, asv_names_host_subset_i in enumerate(asv_names_host):

                abundance_trajectory = read_counts_host[asv_names_host_subset_i_idx,:]

                if sum(abundance_trajectory==0) > 0:
                    continue

                gamma_sampling_model = stats_utils.mle_gamma_sampling(total_abundance, abundance_trajectory)
                mu_start = numpy.mean(abundance_trajectory/total_abundance)
                sigma_start = numpy.std(abundance_trajectory/total_abundance)
                start_params = numpy.asarray([mu_start, sigma_start])
                gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=start_params, bounds= [(0.000001,1), (0.00001,100)], full_output=False, disp=False)

                x_mean, x_std = gamma_sampling_result.params
                
                mle_dict[dataset][host][asv_names_host_subset_i] = {}
                mle_dict[dataset][host][asv_names_host_subset_i]['rel_abundance'] = (abundance_trajectory/total_abundance).tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['days'] = days_host.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['x_mean'] = x_mean
                mle_dict[dataset][host][asv_names_host_subset_i]['x_std'] = x_std


    sys.stderr.write("Saving dictionary...\n")
    with open(mle_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




def plot_deviation_data():

    mle_dict = pickle.load(open(mle_dict_path, "rb"))

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    x_mean_all = []
    for dataset in data_utils.dataset_all:
        for host in mle_dict[dataset].keys():
            for asv in  mle_dict[dataset][host].keys():
                x_mean_all.append(mle_dict[dataset][host][asv]['x_mean'])

    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(x_mean_all), max(x_mean_all)), cmap = plt.get_cmap('Blues'))

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            host_count+=1

            for asv in  mle_dict[dataset][host].keys():
                
                rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                x_mean = mle_dict[dataset][host][asv]['x_mean']
                cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_std']

                #x_mean_log = stats_utils.expected_value_log_gamma(x_mean, cv)
                #log_rel_abundance = numpy.log(rel_abundance)
                
                run_dict = data_utils.calculate_mean_deviation_pattern_data(rel_abundance, x_mean, min_run_length=10, epsilon=0.2*x_mean)
                #run_log_dict = data_utils.calculate_mean_deviation_pattern_data(log_rel_abundance, x_mean, min_run_length=10, epsilon=0.2*x_mean_log)

                if len(run_dict) == 0:
                    continue

                for run_length, run_sojourn in run_dict.items():

                    s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)

 
                    for run_sojourn_j in run_sojourn:

                        #rescaled_run_sojourn_j = run_sojourn_j/x_mean

                        ax.plot(s_range, run_sojourn_j, lw=1, alpha=1, c=cmap.to_rgba(x_mean), ls='-')

                
            ax.set_yscale('log', basey=10)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdeviation_data.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()






#make_mle_dict()

plot_deviation_data()