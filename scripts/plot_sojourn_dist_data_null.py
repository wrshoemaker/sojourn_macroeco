

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

mle_null_dict_path = '%smle_null_dict.pickle' % config.data_directory




def make_mle_null_dict(n_iter=1000):

    mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

    mle_null_dict = {}

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        mle_null_dict[dataset] = {}

        for host in host_all:

            days_run_lengths_all = []
            days_run_lengths_null_all = []

            mle_null_dict[dataset][host] = {}
            mle_null_dict[dataset][host]['otu_stats'] = {}
            mle_null_dict[dataset][host]['prob_sojourn'] = {}

            for key, value in mle_dict[dataset][host].items():
                
                x_mean = value['x_mean']
                x_std = value['x_std']
                rel_abundance = numpy.asarray(value['rel_abundance'])
                days = numpy.asarray(value['days'])
                days_run_lengths = value['days_run_lengths']

                log_rescaled_rel_abundance = numpy.log(rel_abundance/x_mean)
                cv_asv = x_std/x_mean
                expected_value_log_gamma = stats_utils.expected_value_log_gamma(1, cv_asv)

                run_values_subset, run_starts_subseet, run_lengths_subset = data_utils.find_runs((log_rescaled_rel_abundance-expected_value_log_gamma)>0, min_run_length=1)
                run_values_new, run_starts_new, run_lengths_new, days_run_lengths = data_utils.run_lengths_to_days(run_values_subset, run_starts_subseet, run_lengths_subset, days)

                # permute
                mean_days_run_lengths_null_all = []
                for n_iter_i in range(n_iter):
                    log_rescaled_rel_abundance_null = numpy.random.permutation(log_rescaled_rel_abundance)

                    run_values_null, run_starts_null, run_lengths_null = data_utils.find_runs((log_rescaled_rel_abundance_null - expected_value_log_gamma)>0, min_run_length=1)
                    run_values_new_null, run_starts_new_null, run_lengths_new_null, days_run_lengths_null = data_utils.run_lengths_to_days(run_values_null, run_starts_null, run_lengths_null, days)
                    mean_days_run_lengths_null_all.append(numpy.mean(days_run_lengths_null))
                    days_run_lengths_null_all.extend(days_run_lengths_null)


                mean_days_run_lengths_null_all = numpy.sort(mean_days_run_lengths_null_all)
                lower_ci_mean_days_run_lengths_null = mean_days_run_lengths_null_all[int(0.025*n_iter)]
                upper_ci_mean_days_run_lengths_null = mean_days_run_lengths_null_all[int(0.975*n_iter)]
                
                # save mean sojourn time....
                value['mean_sojourn_days'] = numpy.mean(days_run_lengths)
                value['lower_ci_null_mean_sojourn_days'] = lower_ci_mean_days_run_lengths_null
                value['upper_ci_null_mean_sojourn_days'] = upper_ci_mean_days_run_lengths_null
                value['null_mean_sojourn_days'] = numpy.mean(mean_days_run_lengths_null_all)

                mle_null_dict[dataset][host]['otu_stats'][key] = value

                days_run_lengths_all.extend(days_run_lengths)


            days_run_lengths_all = numpy.asarray(days_run_lengths_all)
            x_range = numpy.arange(1, max(days_run_lengths_all)+1)
            days_survival = data_utils.make_survival_dist(days_run_lengths_all, x_range)

            days_run_lengths_null_all = numpy.asarray(days_run_lengths_null_all)
            x_range_null = numpy.arange(1, max(days_run_lengths_null_all)+1)
            days_null_survival = data_utils.make_survival_dist(days_run_lengths_null_all, x_range_null)

            mle_null_dict[dataset][host]['prob_sojourn']['x_range'] = x_range.tolist()
            mle_null_dict[dataset][host]['prob_sojourn']['days_survival'] = days_survival.tolist()

            mle_null_dict[dataset][host]['prob_sojourn']['x_range_null'] = x_range_null.tolist()
            mle_null_dict[dataset][host]['prob_sojourn']['days_null_survival'] = days_null_survival.tolist()


    sys.stderr.write("Saving dictionary...\n")
    with open(mle_null_dict_path, 'wb') as outfile:
        pickle.dump(mle_null_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_dist_w_null():

    mle_null_dict = pickle.load(open(mle_null_dict_path, "rb"))

    n_rows = len(data_utils.dataset_all)
    n_cols = 4
    fig = plt.figure(figsize = (16, 12)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        host_all = list(mle_null_dict[dataset].keys())
        host_all.sort()

        for host_idx, host in enumerate(host_all):

            x_range = mle_null_dict[dataset][host]['prob_sojourn']['x_range']
            days_survival = mle_null_dict[dataset][host]['prob_sojourn']['days_survival']

            x_range_null = mle_null_dict[dataset][host]['prob_sojourn']['x_range_null']
            days_null_survival = mle_null_dict[dataset][host]['prob_sojourn']['days_null_survival']

            x_range = numpy.asarray(x_range)
            days_survival = numpy.asarray(days_survival)

            x_range_null = numpy.asarray(x_range_null)
            days_null_survival = numpy.asarray(days_null_survival)

            ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)
            
            ax.plot(x_range, days_survival, c=plot_utils.host_color_dict[dataset][host], lw=3, linestyle='-', zorder=2, label='Observed')
            ax.plot(x_range_null, days_null_survival, c='k', lw=3, linestyle='-', zorder=2, label='Permutation-based null')

            ax.set_xlim([1, max(x_range)+1])
            ax.set_ylim([min(days_survival), 1])

            ax.set_yscale('log', base=10)

            if host_idx == 0:
                ax.set_ylabel('Fraction of sojourn trajectories ' + r'$\geq \mathcal{T}$', fontsize=9)        
            
            # x-label
            if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
                ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=9)

            if dataset_idx + host_idx == 0:
                ax.legend(loc = 'upper right')



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_dist_data_null.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_mean_sojourn_time_w_null():

    mle_null_dict = pickle.load(open(mle_null_dict_path, "rb"))

    n_rows = len(data_utils.dataset_all)
    n_cols = 4
    fig = plt.figure(figsize = (16, 12)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host_idx, host in enumerate(host_all):

            asv_all = []
            mean_sojourn_days_all = []
            null_mean_sojourn_days_all = []
            lower_ci_null_mean_sojourn_days_all = []
            upper_ci_null_mean_sojourn_days_all = []

            for key, value in mle_null_dict[dataset][host]['otu_stats'].items():

                asv_all.append(key)
                mean_sojourn_days_all.append(value['mean_sojourn_days'])
                null_mean_sojourn_days_all.append(value['null_mean_sojourn_days'])
                lower_ci_null_mean_sojourn_days_all.append(value['lower_ci_null_mean_sojourn_days'])
                upper_ci_null_mean_sojourn_days_all.append(value['upper_ci_null_mean_sojourn_days'])

            asv_all = numpy.asarray(asv_all)
            mean_sojourn_days_all = numpy.asarray(mean_sojourn_days_all)
            null_mean_sojourn_days_all = numpy.asarray(null_mean_sojourn_days_all)
            lower_ci_null_mean_sojourn_days_all = numpy.asarray(lower_ci_null_mean_sojourn_days_all)
            upper_ci_null_mean_sojourn_days_all = numpy.asarray(upper_ci_null_mean_sojourn_days_all)

            sort_idx = numpy.argsort(mean_sojourn_days_all)

            asv_all = asv_all[sort_idx]
            mean_sojourn_days_all = mean_sojourn_days_all[sort_idx]
            null_mean_sojourn_days_all = null_mean_sojourn_days_all[sort_idx]
            lower_ci_null_mean_sojourn_days_all = lower_ci_null_mean_sojourn_days_all[sort_idx]
            upper_ci_null_mean_sojourn_days_all = upper_ci_null_mean_sojourn_days_all[sort_idx]

            ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            for i_idx in range(len(asv_all)):

                ax.scatter(mean_sojourn_days_all[i_idx], i_idx, s=10, c='dodgerblue', zorder=2)

                uppper_null_i = upper_ci_null_mean_sojourn_days_all[i_idx]
                lower_null_i = lower_ci_null_mean_sojourn_days_all[i_idx]
                mean_null_i = null_mean_sojourn_days_all[i_idx]
                ax.errorbar(mean_null_i, i_idx, xerr=[[mean_null_i - lower_null_i], [uppper_null_i - mean_null_i]], mfc = 'white', zorder=1, c = 'k')


            ax.set_xlabel('Mean sojourn time (days), ' + r'$\bar{T}$', fontsize=9)
            ax.set_ylabel('ASVs', fontsize=9)

            host_count+=1


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%smean_sojourn_time_w_null.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    #make_mle_null_dict()
    plot_dist_w_null()
