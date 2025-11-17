import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
from collections import Counter
import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma
from scipy import integrate
from scipy.optimize import minimize_scalar


#import functools
#import operator

import data_utils
import plot_utils
import stats_utils
import simulation_utils
import theory_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors, colorbar

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

max_sojourn_time = theory_utils.max_sojourn_time


def make_null_gamma_sojourn_time_dist(max_sojourn_time=max_sojourn_time):

    n_obs_per_dist = []
    days_run_lengths_all = []

    sojourn_time_range = numpy.arange(1, max_sojourn_time+1)

    sojourn_time_pdf_gamma_null_fin_all = []
    n_obs_all = []

    sojourn_time_pdf_all = []
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
                days_run_values = mle_dict[dataset][host][asv]['days_run_values']

                run_lengths = numpy.asarray(run_lengths)
                days_run_lengths = numpy.asarray(days_run_lengths)
                days_run_values = numpy.asarray(days_run_values)

                #days_run_lengths_pos = days_run_lengths[days_run_values==True]

                x_mean = mle_dict[dataset][host][asv]['x_mean']
                x_cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_mean']
                #x_beta = (1/x_cv)**2
                #expected_log_rescaled_x = stats_utils.expected_value_log_gamma(1, x_cv)
                #expected_log_rescaled_x = stats_utils.expected_value_rescaled_log_gamma(x_cv)

                # (1-p)
                #cdf_value = stats.loggamma.cdf(expected_log_rescaled_x, c=x_beta, scale=1/x_beta)
                cdf_value = stats_utils.right_tail_log_rescaled_gamma(x_mean, x_cv)[0]
                range_run_lengths = numpy.arange(1, n_samples+1)
                numerator = ((cdf_value**range_run_lengths) + ((1-cdf_value)**range_run_lengths))

                # 1) rescale assuming infinite samples
                #z_inf = (cdf_value/(1-cdf_value)) + ((1-cdf_value)/cdf_value)
                #sojourn_time_pdf_gamma_null_inf = range_run_lengths/z_inf


                # 2) rescale assuming finite samples set by empirical number
                sojourn_time_pdf_gamma_null_fin = numerator/sum(numerator)

                #print(sum(sojourn_time_pdf_gamma_null_fin), sum(sojourn_time_pdf_gamma_null_inf))
                #print(sojourn_time_pdf_gamma_null_fin[0:15])

                #sojourn_time_pdf = ((cdf_value**sojourn_time_range) + ((1-cdf_value)**sojourn_time_range))/z
                #sojourn_time_pdf = ((cdf_value**sojourn_time_range))/z
                #sojourn_time_pdf_gamma_null = (cdf_value**run_lengths)

                #print('n samples', run_lengths)
                #print('days', days_run_lengths)
                

                #sojourn_time_pdf = geom
                sojourn_time_pdf_gamma_null_fin_all.append(sojourn_time_pdf_gamma_null_fin)
                n_obs_all.append(len(run_lengths))
                #days_run_lengths_all.extend(days_run_lengths.tolist())
                days_run_lengths_all.append(days_run_lengths)


    # calculate mixture
    n_obs_all = numpy.asarray(n_obs_all)
    weights_all = n_obs_all/sum(n_obs_all)
    max_range = max([len(pdf) for pdf in sojourn_time_pdf_gamma_null_fin_all])
    n_samples_range = numpy.arange(1, max_range + 1)
    mixture_pdf_null = numpy.zeros_like(n_samples_range, dtype=float)
    for pdf, w in zip(sojourn_time_pdf_gamma_null_fin_all, weights_all):
        mixture_pdf_null[:len(pdf)] += w * pdf

    mixture_pdf_null = mixture_pdf_null/sum(mixture_pdf_null)


    #sojourn_time_pdf_gamma_null_fin_all = numpy.stack(sojourn_time_pdf_gamma_null_fin_all, axis=0)
    ##n_obs_per_dist = numpy.asarray(n_obs_per_dist)
    ##weights_per_dist = n_obs_per_dist/sum(n_obs_per_dist)
    #mixture_dist = numpy.sum(sojourn_time_pdf_all * weights_per_dist[:, numpy.newaxis], axis=0)
    # get empirical distribution
    #days_run_lengths_dict = dict(Counter(days_run_lengths_all))
    #sojourn_data_range = numpy.sort(list(days_run_lengths_dict.keys()))
    #sojourn_data_range.sort()
    #sojourn_data_pdf = numpy.asarray([days_run_lengths_dict[s] for s in sojourn_data_range])
    #sojourn_data_pdf = sojourn_data_pdf/sum(sojourn_data_pdf)
    
    # get the weights for the mixture empirical distribution

    pdf_obs_all = []
    weights_obs_all = []

    sojourn_obs_range = numpy.unique(numpy.concatenate(days_run_lengths_all))
    mixture_pdf = numpy.zeros_like(sojourn_obs_range, dtype=float)
    
    for arr in days_run_lengths_all:
        unique, counts = numpy.unique(arr, return_counts=True)
        pdf = counts/sum(counts)
        indices = numpy.searchsorted(sojourn_obs_range, unique)
        mixture_pdf[indices] += len(arr) * pdf

    mixture_pdf = mixture_pdf/sum(mixture_pdf)
    
    # match empirical range to prediction
    cutoff_idx = numpy.where(sojourn_time_range == max(sojourn_obs_range))[0][0]
    sojourn_null_range = sojourn_time_range[:cutoff_idx]
    sojourn_null_pdf = mixture_pdf_null[:cutoff_idx]


    return sojourn_obs_range, mixture_pdf, sojourn_null_range, sojourn_null_pdf
    



def make_ou_sojourn_time_dist(max_sojourn_time=max_sojourn_time, tau_0=1, tau_1=0, y_0=data_utils.epsilon_fract_data, normalize=False):

    # data_utils.epsilon_fract_data

    #if prob_t_equal_1 != None:
    #else:
    #    alpha

    n_obs_per_dist = []

    sojourn_time_range = numpy.arange(1, max_sojourn_time+1)
    sojourn_time_pdf_all = []

    tau_new_all = []
    for dataset in data_utils.dataset_all:
    
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            for asv in  mle_dict[dataset][host].keys():
                
                days_run_lengths = mle_dict[dataset][host][asv]['days_run_lengths']
                days_run_values = mle_dict[dataset][host][asv]['days_run_values']
                run_starts = mle_dict[dataset][host][asv]['run_starts']
                rel_abundance = mle_dict[dataset][host][asv]['rel_abundance']
                #max_possible_sojourn_time = mle_dict[dataset][host][asv]['max_possible_sojourn_time']

                #y_0 = mle_dict[dataset][host][asv]['rel_abundance'][0]
                days_run_lengths = numpy.asarray(days_run_lengths)
                days_run_values = numpy.asarray(days_run_values)
                run_starts = numpy.asarray(run_starts)
                rel_abundance = numpy.asarray(rel_abundance)

                #days_run_lengths_pos = days_run_lengths[days_run_values==True]
                run_starts_pos = run_starts[days_run_values==True]

                x_mean = mle_dict[dataset][host][asv]['x_mean']
                x_cv = mle_dict[dataset][host][asv]['x_std']/x_mean

                k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
                #expected_log_recaled_x = stats_utils.expected_value_log_gamma(1, x_cv)

                tau_new = tau_0 + tau_1*sigma
                tau_new = 10
                sojourn_time_pdf = theory_utils.predict_sojourn_dist_ou(max_sojourn_time, sigma, tau_new, y_0)

                sojourn_time_pdf_all.append(sojourn_time_pdf)
                n_obs_per_dist.append(len(days_run_lengths))

                tau_new_all.append(tau_new)



    # calculate mixture
    sojourn_time_pdf_all = numpy.stack(sojourn_time_pdf_all, axis=0)
    n_obs_per_dist = numpy.asarray(n_obs_per_dist)
    weights_per_dist = n_obs_per_dist/sum(n_obs_per_dist)
    mixture_dist = numpy.sum(sojourn_time_pdf_all * weights_per_dist[:, numpy.newaxis], axis=0)

    if normalize == True:
        mixture_dist = mixture_dist/sum(mixture_dist)

    return sojourn_time_range, mixture_dist





def identify_ml_tau(max_sojourn_time=max_sojourn_time):

    sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = make_null_gamma_sojourn_time_dist(max_sojourn_time=max_sojourn_time)

    def negative_log_likelihood(tau):
        predicted = make_ou_sojourn_time_dist(tau_0=tau)[1]
        predicted_same_days = predicted[(sojourn_data_range-1)]
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-12
        return -numpy.sum(sojourn_data_pdf * numpy.log(predicted_same_days + epsilon))
    

    result = minimize_scalar(negative_log_likelihood, bounds=(0.01, 10), method='bounded')

    best_tau = result.x

    print(best_tau)



def make_null_perm_sojourn_time_dist():

    mle_null_dict_path = '%smle_null_dict.pickle' % config.data_directory
    mle_null_dict = pickle.load(open(mle_null_dict_path, "rb"))

    x_range_pdf_null_all = []
    days_pdf_null_all = []
    n_obs_all = []

    for dataset_idx, dataset in enumerate(data_utils.dataset_all):

        sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            x_range_pdf_null = numpy.asarray(mle_null_dict[dataset][host]['prob_sojourn']['x_range_pdf_null'])
            days_pdf_null = numpy.asarray(mle_null_dict[dataset][host]['prob_sojourn']['days_pdf_null'])
            n_obs = mle_null_dict[dataset][host]['prob_sojourn']['n_obs']

            x_range_pdf_null_all.append(x_range_pdf_null)
            days_pdf_null_all.append(days_pdf_null)
            n_obs_all.append(n_obs)


    weights = numpy.array(n_obs_all, dtype=float)
    weights = weights/sum(weights)

    all_rv = numpy.unique(numpy.concatenate(x_range_pdf_null_all))
    x_range_pdf_null_mix = numpy.sort(all_rv)

    mixture_pdf = numpy.zeros_like(x_range_pdf_null_mix, dtype=float)

    for pdf, rv, w in zip(days_pdf_null_all, x_range_pdf_null_all, weights):
        indices = numpy.searchsorted(x_range_pdf_null_mix, rv)
        mixture_pdf[indices] += w * pdf

    mixture_pdf = mixture_pdf/sum(mixture_pdf)

    return x_range_pdf_null_mix, mixture_pdf
            





def plot_sojourn_time_mix_dist(plot_gamma_null=False):

    #colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}

    fig, ax = plt.subplots(figsize=(5,4))

    sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = make_null_gamma_sojourn_time_dist()
    #sojourn_perm_range, sojourn_perm_pdf = make_null_perm_sojourn_time_dist()

    mle_null_dict_path = '%smle_null_dict.pickle' % config.data_directory
    mle_null_dict = pickle.load(open(mle_null_dict_path, "rb"))

    sojourn_perm_range = numpy.asarray(mle_null_dict['mixture']['x_range_pdf_null'])
    sojourn_perm_pdf = numpy.asarray(mle_null_dict['mixture']['days_pdf_null'])

    # mean sigma = 0.6033768396901017
    # = 0.6033768396901017
    tau_3 = 3
    #tau_2 = 2
    #tau_1 = tau_0/mean_sigma
    
    #target_asv_color = '#eb5900'
    # 87CEEB
    #ax.plot(sojourn_data_range, sojourn_data_pdf, c='#eb5900', lw=4, ls='-', label='Data')
    ax.plot(sojourn_data_range, 1-numpy.cumsum(sojourn_data_pdf), c='#eb5900', lw=4, ls='-', label='Data')

    #if plot_gamma_null == True:
    # analytic gamma null
    #ax.plot(sojourn_null_range, sojourn_null_pdf, c='k', lw=4, ls=':', label=r'$\tau \ll \delta t $' + ' (gamma)')
    ax.plot(sojourn_null_range, 1-numpy.cumsum(sojourn_null_pdf), c='k', lw=4, ls=':', label=r'$\tau \ll \delta t $' + ' (gamma)')
    sojourn_prediction_label = r'$\tau \sim \mathcal{O}(\delta t)$'
    plot_gamma_null_label = '_w_gamma'

    # permutation-based null
    #ax.plot(sojourn_perm_range, sojourn_perm_pdf, c='k', lw=4, ls='--', label="Time-permuted null")
    ax.plot(sojourn_perm_range, 1-numpy.cumsum(sojourn_perm_pdf), c='k', lw=4, ls='--', label="Time-permuted null")
    sojourn_prediction_label = r'$\tau \sim \mathcal{O}(\delta t)$'

    

    #else:
    #    sojourn_prediction_label = 'Prediction'
    #    plot_gamma_null_label = ''

    # #FF6347
    #color_tau_all = ['lightskyblue', 'dodgerblue', 'royalblue']
    #tau_all = [1, 2, 3]
    #for tau_idx, tau in enumerate(tau_all):
    #    sojourn_time_range_tau, mixture_dist_tau = make_ou_sojourn_time_dist(tau=tau)
    #    ax.plot(sojourn_time_range_tau, mixture_dist_tau, c=color_tau_all[tau_idx], lw=2, ls='-', label=r'$\tau = $' + str(tau) + ' (OU)')
    #ax.plot(sojourn_time_range_tau, mixture_dist_tau, c='dodgerblue', lw=2, ls='-', label='OU, ' + r'$\tau_{i} \propto \sigma_{i}$')

    # 2.715195778605458
    #sojourn_time_range_tau_2, mixture_dist_tau_2 = make_ou_sojourn_time_dist(tau_0=tau_3, tau_1=0, normalize=True)
    #ax.plot(sojourn_time_range_tau_2, mixture_dist_tau_2, c='k', lw=4, ls=':', label=r'$\tau = $' + str(tau_2))

    sojourn_time_range_tau_2, mixture_dist_tau_2 = make_ou_sojourn_time_dist(tau_0=tau_3, tau_1=0, normalize=True)
    #ax.plot(sojourn_time_range_tau_2, mixture_dist_tau_2, c='k', lw=4, ls='-', label=sojourn_prediction_label)
    ax.plot(sojourn_time_range_tau_2, 1-numpy.cumsum(mixture_dist_tau_2), c='k', lw=4, ls='-', label=sojourn_prediction_label)


    #sojourn_time_range_tau_3, mixture_dist_tau_3 = make_ou_sojourn_time_dist(tau_0=tau_2, tau_1=0, normalize=True)
    #ax.plot(sojourn_time_range_tau_3, mixture_dist_tau_3, c='k', lw=4, ls='--', label=r'$\tau = $' + str(tau_3))


    ax.set_xlim([1, max(sojourn_data_range)])
    #ax.set_ylim([min(sojourn_data_pdf), 1])
    ax.set_ylim([min(sojourn_data_pdf), 1])

    #ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=17)
    ax.set_ylabel("Probability density, " + r'$P(\mathcal{T} \,)$', fontsize=17)
    ax.legend(loc='upper right', fontsize=13)

    

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_data_mix%s.png" % (config.analysis_directory, plot_gamma_null_label)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




#make_null_gamma_sojourn_time_dist()
#identify_ml_tau()
#plot_sojourn_time_mix_dist(plot_gamma_null=False)
plot_sojourn_time_mix_dist(plot_gamma_null=True)
#make_ou_sojourn_time_dist(max_sojourn_time=100, tau=1)


