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
from scipy import integrate
import data_utils
import plot_utils
import stats_utils
import pickle

n_iter=10

#colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}

numpy.random.seed(123456789)

res_ret_dict_path = '%sres_ret_dict.pickle' % config.data_directory


res_color = '#87CEEB'
ret_color = '#FF6347'
latex_label_dict = {True: 'Residence time, ' + r'$t_{\mathrm{res}}$', False: 'Return time, ' + r'$t_{\mathrm{ret}}$'}
str_dict = {True: 'residence', False: 'return'}




def identify_runs(x, days, residence=True):

    x = numpy.asarray(x)
    days = numpy.asarray(days)

    if residence == True:
        mask = x > 0
    else:
        mask = x == 0
        # return

    diff = numpy.diff(mask.astype(int))
    starts = numpy.where(diff == 1)[0] + 1
    ends = numpy.where(diff == -1)[0] + 1

    if mask[0]:
        starts = numpy.r_[0, starts]
    if mask[-1]:
        ends = numpy.r_[ends, len(x)]

    run_days = numpy.array([days[e - 1] - days[s] + 1 for s, e in zip(starts, ends)])

    return run_days




#n_rows = 10
#n_cols = 2
#fig = plt.figure(figsize = (4, 20)) #
#fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
# null (no dynamics) PDFs can be derived for each as geometric distributions

def make_res_ret_dict():

    res_ret_dict = {}

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):
        
        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, 'gut')
        host_all = list(set(host_status))
        host_all.sort()

        res_ret_dict[dataset] = {}

        for host_idx, host in enumerate(host_all):

            sys.stderr.write("Analyzing host %s.....\n" % host)
    
            read_counts_host, days_host, asv_names_host = data_utils.subset_s_by_s_by_host(read_counts, host_status, days, asv_names, host)

            res_ret_dict[dataset][host] = {}

            for res_bool in [True, False]:

                res_days = []
                res_days_null = []

                for asv_names_host_subset_i_idx, asv_names_host_subset_i in enumerate(asv_names_host):

                    abundance_trajectory = read_counts_host[asv_names_host_subset_i_idx,:]

                    # ignore ASVs with occupancy == 1
                    # or
                    # < 10 non-zero abundances
                    if (sum(abundance_trajectory==0) == 0) or (sum(abundance_trajectory>0) < 10):
                        continue

                    res_days.append(identify_runs(abundance_trajectory, days_host, residence=res_bool))

                    for i in range(n_iter):

                        abundance_trajectory_null = numpy.random.permutation(abundance_trajectory)
                        res_days_null.append(identify_runs(abundance_trajectory_null, days_host, residence=res_bool))


                res_days = numpy.concatenate(res_days, axis=0)
                res_days_range, res_days_pdf = stats_utils.get_pdf_from_counts(res_days)

                res_days_null = numpy.concatenate(res_days_null, axis=0)
                res_days_range_null, res_days_pdf_null = stats_utils.get_pdf_from_counts(res_days_null)

                res_ret_dict[dataset][host][res_bool] = {}
                res_ret_dict[dataset][host][res_bool]['x_range_pdf'] = res_days_range.tolist()
                res_ret_dict[dataset][host][res_bool]['days_pdf'] = res_days_pdf.tolist()
                
                res_ret_dict[dataset][host][res_bool]['x_range_pdf_null'] = res_days_range_null.tolist()
                res_ret_dict[dataset][host][res_bool]['days_pdf_null'] = res_days_pdf_null.tolist()



    sys.stderr.write("Saving dictionary...\n")
    with open(res_ret_dict_path, 'wb') as outfile:
        pickle.dump(res_ret_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




def plot_res_ret_time(res_bool=True):

    res_ret_dict = pickle.load(open(res_ret_dict_path, "rb"))

    n_rows = len(data_utils.dataset_all)
    n_cols = 4
    fig = plt.figure(figsize = (16, 12)) #
    #fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    fig.suptitle(latex_label_dict[res_bool], fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down

    #row_idx = 0
    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        
        read_counts, host_status, days, asv_names = data_utils.get_dada2_data(dataset, 'gut')
        host_all = list(set(host_status))
        host_all.sort()

        for host_idx, host in enumerate(host_all):

            sys.stderr.write("Analyzing host %s.....\n" % host)
            ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))
            ax.set_title('%s, %s' % (dataset, host), fontsize=11)

            x_range_pdf = numpy.asarray(res_ret_dict[dataset][host][res_bool]['x_range_pdf'])
            days_pdf = numpy.asarray(res_ret_dict[dataset][host][res_bool]['days_pdf'])
                
            x_range_pdf_null = numpy.asarray(res_ret_dict[dataset][host][res_bool]['x_range_pdf_null'])
            days_pdf_null = numpy.asarray(res_ret_dict[dataset][host][res_bool]['days_pdf_null'])

            ax.plot(x_range_pdf, days_pdf, c=plot_utils.host_color_dict[dataset][host], lw=2, ls='-', alpha=1, label='Observed')
            ax.plot(x_range_pdf_null, days_pdf_null, c='k', lw=2, ls='-', alpha=1, label='Permutation-based null')


            ax.set_xlim([1, max(x_range_pdf)])
            ax.set_ylim([min(days_pdf), 1])

            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)

            ax.xaxis.set_tick_params(labelsize=7)
            ax.yaxis.set_tick_params(labelsize=7)


            if host_idx == 0:
                ax.set_ylabel("Probability density", fontsize=10)        
            
            # x-label
            if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
                ax.set_xlabel(latex_label_dict[res_bool], fontsize=10)

            if dataset_idx + host_idx == 0:
                ax.legend(loc = 'upper right')
            
            #row_idx+=1        

            #common_ids = numpy.intersect1d(res_days_range, res_days_range_null)
            #mask = numpy.isin(res_days_range, common_ids)
            #mask_null = numpy.isin(res_days_range_null, common_ids)
            #res_days_pdf_shared = res_days_pdf[mask]
            #res_days_pdf_null_shared = res_days_pdf_null[mask_null]


            #print(stats_utils.js_div(res_days_pdf_shared, res_days_pdf_null_shared))


    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    fig_name = "%s%s_time.png" % (config.analysis_directory, str_dict[res_bool])
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




if __name__ == "__main__":

    print("Running...")

    #make_res_ret_dict()

    #plot_res_ret_time(res_bool=True)
    plot_res_ret_time(res_bool=False)