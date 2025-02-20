
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




def make_mle_dict(epsilon_fract=0.01, min_run_length=10):

    mle_dict = {}
    mle_dict['params'] = {}
    mle_dict['params']['epsilon_fract'] = epsilon_fract
    mle_dict['params']['min_run_length'] = min_run_length

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

                # ignore ASVs with occupancy < 1
                if sum(abundance_trajectory==0) > 0:
                    continue

                rel_abundance_trajectory = abundance_trajectory/total_abundance

                # gamma MLE paramss
                gamma_sampling_model = stats_utils.mle_gamma_sampling(total_abundance, abundance_trajectory)
                mu_start = numpy.mean(abundance_trajectory/total_abundance)
                sigma_start = numpy.std(abundance_trajectory/total_abundance)
                start_params = numpy.asarray([mu_start, sigma_start])
                gamma_sampling_result = gamma_sampling_model.fit(method="lbfgs", start_params=start_params, bounds= [(0.000001,1), (0.00001,100)], full_output=False, disp=False)
                x_mean, x_std = gamma_sampling_result.params

                # get sojourn times for runs of all lengths
                run_values, run_starts, run_lengths = data_utils.find_runs((rel_abundance_trajectory - x_mean)>0, min_run_length=1)
                days_run_lengths = []
                for run_j_idx in range(len(run_values)):
                    
                    days_run_j = days_host[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]]

                    # only one observation, skip
                    if len(days_run_j) == 1:
                        continue

                    days_run_lengths.append(int(days_run_j[-1] - days_run_j[0]))



                if len(days_run_lengths) == 0:
                    continue

                mle_dict[dataset][host][asv_names_host_subset_i] = {}
                mle_dict[dataset][host][asv_names_host_subset_i]['rel_abundance'] = rel_abundance_trajectory.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['days'] = days_host.tolist()
                mle_dict[dataset][host][asv_names_host_subset_i]['x_mean'] = x_mean
                mle_dict[dataset][host][asv_names_host_subset_i]['x_std'] = x_std
                mle_dict[dataset][host][asv_names_host_subset_i]['days_run_lengths'] = days_run_lengths

                #for run_length, run_sojourn in run_dict.items():
                run_dict = data_utils.calculate_deviation_pattern_data(rel_abundance_trajectory, x_mean, min_run_length=min_run_length, epsilon=epsilon_fract*x_mean, return_array=False)

                if len(run_dict) == 0:
                    mle_dict[dataset][host][asv_names_host_subset_i]['run_dict'] = None
                else:
                    mle_dict[dataset][host][asv_names_host_subset_i]['run_dict'] = run_dict


                # add runs where the deviation can be examined

    sys.stderr.write("Saving dictionary...\n")
    with open(mle_dict_path, 'wb') as outfile:
        pickle.dump(mle_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_sojourn_vs_norm():

    mle_dict = pickle.load(open(mle_dict_path, "rb"))

    sojourn_time_all = []
    norm_all = []

    for dataset in mle_dict.keys():

        for host in mle_dict[dataset].keys():

            for asv in mle_dict[dataset][host]:

                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict == None:
                    continue

                for sojourn_time, sojourn_trajectory_list_nested in run_dict.items():

                    rescald_time_range = numpy.linspace(0, 1, num=sojourn_time, endpoint=True)

                    for sojourn_trajectory_list in sojourn_trajectory_list_nested:

                        #sojourn_trajectory_list = numpy.asarray(sojourn_trajectory_list)/mle_dict[dataset][host][asv]['x_mean']
                        sojourn_trajectory_list = numpy.asarray(sojourn_trajectory_list)

                        to_keep_idx = sojourn_trajectory_list > 0

                        sojourn_trajectory_list = sojourn_trajectory_list[to_keep_idx]
                        rescald_time_range_ = rescald_time_range[to_keep_idx]

                        norm = stats_utils.estimate_normalization_constant(rescald_time_range_, sojourn_trajectory_list)
                        
                        sojourn_time_all.append(sojourn_time)
                        norm_all.append(norm)



    sojourn_time_all = numpy.asarray(sojourn_time_all)
    norm_all = numpy.asarray(norm_all)


    slope, intercept = stats_utils.log_log_regression(sojourn_time_all, norm_all)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.scatter(sojourn_time_all, norm_all, c='k', alpha=0.6, s=10)

    x_log10_range =  numpy.linspace(min(numpy.log10(sojourn_time_all)) , max(numpy.log10(sojourn_time_all)) , 10000)
    y_log10_fit_range = (slope*x_log10_range + intercept)

    ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")

    ax.text(0.5, 0.1, 'Exponent = %.2f' % slope, fontsize=13, transform=ax.transAxes)


    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel('Sojourn time, ' + r'$T$', fontsize=12)
    ax.set_ylabel("Normalization constant, " + r'$N(T)$', fontsize=12)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_vs_norm.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_deviation_data(sojourn_time, min_x_mean = 0):

    mle_dict = pickle.load(open(mle_dict_path, "rb"))

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    host_count = 0

    x_mean_all = []
    for dataset in data_utils.dataset_all:
        for host in mle_dict[dataset].keys():
            for asv in  mle_dict[dataset][host].keys():
                
                if mle_dict[dataset][host][asv]['x_mean'] <= min_x_mean:
                    continue

                x_mean_all.append(mle_dict[dataset][host][asv]['x_mean'])

    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(x_mean_all), max(x_mean_all)), cmap = plt.get_cmap('Blues'))


    run_length_all = []

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            sys.stderr.write("Analyzing host %s.....\n" % host)

            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            host_count+=1

            ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=9)
            ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=9)


            for asv in  mle_dict[dataset][host].keys():
                
                rel_abundance = numpy.asarray(mle_dict[dataset][host][asv]['rel_abundance'])
                x_mean = mle_dict[dataset][host][asv]['x_mean']
                #cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_mean']

                if x_mean <= min_x_mean:
                    continue

                
                run_dict = mle_dict[dataset][host][asv]['run_dict']

                if run_dict is None:
                    continue

                run_length_all.extend(list(run_dict.keys()))

                for run_length, run_sojourn in run_dict.items():

                    if run_length != sojourn_time:
                        continue

                    s_range = numpy.linspace(0, 1, num=run_length, endpoint=True)
 
                    for run_sojourn_j in run_sojourn:

                        run_sojourn_j = numpy.asarray(run_sojourn_j)

                        #print(len(run_sojourn_j))
                        run_sojourn_j = numpy.absolute(run_sojourn_j)
                        
                        # ensure no negative values
                        to_plot_idx = (run_sojourn_j>0)
                        s_range_to_plot = s_range[to_plot_idx]
                        run_sojourn_j_to_plot = run_sojourn_j[to_plot_idx]
                        rescaled_run_sojourn_j_to_plot = run_sojourn_j_to_plot/x_mean
                        ax.plot(s_range_to_plot, rescaled_run_sojourn_j_to_plot, lw=1, alpha=1, c=cmap.to_rgba(x_mean), ls='-')

                
            ax.set_yscale('log', base=10)
            ax.set_xlim([0,1])


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdeviation_data_rescaled/deviation_data_rescaled_%d.png" % (config.analysis_directory, sojourn_time)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_dist_run_lengths():

    fig = plt.figure(figsize = (8, 20)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    mle_dict = pickle.load(open(mle_dict_path, "rb"))

    host_count = 0

    for dataset in data_utils.dataset_all:

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)

        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:


            days_run_lengths_all = []
            cv_all = []
            for key, value in mle_dict[dataset][host].items():

                days_run_lengths_all.extend(value['days_run_lengths'])

                cv_all.append(value['x_std']/value['x_mean'])

                
            days_run_lengths_all = numpy.asarray(days_run_lengths_all)

            print(dataset, host, numpy.mean(cv_all))


            ax = plt.subplot2grid((5, 2), ax_idx_all[host_count])
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            #days_run_lengths_all_pos = days_run_lengths_all[run_values_all==True]
            #days_run_lengths_all_neg = days_run_lengths_all[run_values_all==False]

            ax.hist(numpy.log10(days_run_lengths_all), bins=9, density=True, lw=3, histtype='step', fill=False, color='dodgerblue')

            #ax.hist(days_run_lengths_all_pos, bins=10, density=True, histtype='step', fill=False, color='b')
            #ax.hist(days_run_lengths_all_neg, bins=10, density=True, histtype='step', fill=False, color='r')

            ax.set_xlabel("Sojourn time (days)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)

            ax.set_yscale('log', base=10)
            #ax.set_xlim([1, 200])

            ax.set_xlim([0, numpy.log10(200)])


            host_count+=1


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%sdist_run_lengths.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





if __name__ == "__main__":

    print("Running...")

    plot_sojourn_vs_norm()


    #plot_dist_run_lengths()

#sojourn_all = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 43, 44, 45, 47, 50, 51, 60, 73]

#for s in sojourn_all:
#    plot_deviation_data(s)

#make_mle_dict(epsilon_fract=0.01)
#plot_deviation_data(12)
