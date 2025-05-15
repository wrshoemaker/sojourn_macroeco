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

    sojourn_time_pdf_all = []
    for dataset in data_utils.dataset_all:
    
        host_all = list(mle_dict[dataset].keys())
        host_all.sort()

        for host in host_all:

            for asv in  mle_dict[dataset][host].keys():
                
                days_run_lengths = mle_dict[dataset][host][asv]['days_run_lengths']
                days_run_values = mle_dict[dataset][host][asv]['days_run_values']

                days_run_lengths = numpy.asarray(days_run_lengths)
                days_run_values = numpy.asarray(days_run_values)

                days_run_lengths_pos = days_run_lengths[days_run_values==True]

                x_cv = mle_dict[dataset][host][asv]['x_std']/mle_dict[dataset][host][asv]['x_mean']
                x_beta = (1/x_cv)**2
                expected_log_rescaled_x = stats_utils.expected_value_log_gamma(1, x_cv)

                # (1-p)
                cdf_value = stats.loggamma.cdf(expected_log_rescaled_x, c=x_beta, scale=1/x_beta)
                #z = (cdf_value/(1-cdf_value)) + ((1-cdf_value)/cdf_value)
                # normalization constant
                z = (cdf_value/(1-cdf_value))

                #sojourn_time_pdf = ((cdf_value**sojourn_time_range) + ((1-cdf_value)**sojourn_time_range))/z
                sojourn_time_pdf = ((cdf_value**sojourn_time_range))/z
                #sojourn_time_pdf = geom
                sojourn_time_pdf_all.append(sojourn_time_pdf)
                n_obs_per_dist.append(len(days_run_lengths_pos))
                days_run_lengths_all.extend(days_run_lengths_pos.tolist())


    # calculate mixture
    sojourn_time_pdf_all = numpy.stack(sojourn_time_pdf_all, axis=0)
    n_obs_per_dist = numpy.asarray(n_obs_per_dist)
    weights_per_dist = n_obs_per_dist/sum(n_obs_per_dist)
    mixture_dist = numpy.sum(sojourn_time_pdf_all * weights_per_dist[:, numpy.newaxis], axis=0)

    # get empirical distribution
    days_run_lengths_dict = dict(Counter(days_run_lengths_all))
    sojourn_data_range = numpy.sort(list(days_run_lengths_dict.keys()))
    #sojourn_data_range.sort()
    sojourn_data_pdf = numpy.asarray([days_run_lengths_dict[s] for s in sojourn_data_range])
    sojourn_data_pdf = sojourn_data_pdf/sum(sojourn_data_pdf)

    # match empirical range to prediction
    cutoff_idx = numpy.where(sojourn_time_range == max(sojourn_data_range))[0][0]
    sojourn_null_range = sojourn_time_range[:cutoff_idx]
    sojourn_null_pdf = mixture_dist[:cutoff_idx]


    return sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf
    



def make_ou_sojourn_time_dist(max_sojourn_time=max_sojourn_time, tau_0=1, tau_1=0, y_0=data_utils.epsilon_fract_data):

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

                #y_0 = mle_dict[dataset][host][asv]['rel_abundance'][0]
                days_run_lengths = numpy.asarray(days_run_lengths)
                days_run_values = numpy.asarray(days_run_values)
                run_starts = numpy.asarray(run_starts)
                rel_abundance = numpy.asarray(rel_abundance)

                days_run_lengths_pos = days_run_lengths[days_run_values==True]
                run_starts_pos = run_starts[days_run_values==True]

                #print(rel_abundance[run_starts_pos[0]])

                x_mean = mle_dict[dataset][host][asv]['x_mean']
                x_cv = mle_dict[dataset][host][asv]['x_std']/x_mean

                k, sigma, m, phi = simulation_utils.calculate_stationary_params_from_moments(x_mean, x_cv)
                #expected_log_recaled_x = stats_utils.expected_value_log_gamma(1, x_cv)

                tau_new = tau_0 + tau_1*sigma
                sojourn_time_pdf = theory_utils.predict_sojourn_dist_ou(max_sojourn_time, sigma, tau_new, y_0)

                sojourn_time_pdf_all.append(sojourn_time_pdf)
                n_obs_per_dist.append(len(days_run_lengths_pos))

                tau_new_all.append(tau_new)


    # calculate mixture
    sojourn_time_pdf_all = numpy.stack(sojourn_time_pdf_all, axis=0)
    n_obs_per_dist = numpy.asarray(n_obs_per_dist)
    weights_per_dist = n_obs_per_dist/sum(n_obs_per_dist)
    mixture_dist = numpy.sum(sojourn_time_pdf_all * weights_per_dist[:, numpy.newaxis], axis=0)

    # normalize probability to sum to one
    #mixture_dist = mixture_dist/sum(mixture_dist)


    return sojourn_time_range, mixture_dist





def identify_ml_tau(max_sojourn_time=max_sojourn_time):

    sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = make_null_gamma_sojourn_time_dist(max_sojourn_time=max_sojourn_time)

    print(sojourn_data_range)
    def negative_log_likelihood(tau):
        predicted = make_ou_sojourn_time_dist(tau_0=tau)[1]
        predicted_same_days = predicted[(sojourn_data_range-1)]
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-12
        return -numpy.sum(sojourn_data_pdf * numpy.log(predicted_same_days + epsilon))
    

    result = minimize_scalar(negative_log_likelihood, bounds=(0.01, 10), method='bounded')

    best_tau = result.x

    print(best_tau)





def plot_sojourn_time_mix_dist():

    #colors_dict = {'0':'#87CEEB', '1': '#FFA500', '2':'#FF6347'}

    fig, ax = plt.subplots(figsize=(4,4))

    sojourn_data_range, sojourn_data_pdf, sojourn_null_range, sojourn_null_pdf = make_null_gamma_sojourn_time_dist()

    # mean sigma = 0.6033768396901017
    mean_sigma = 0.6033768396901017
    tau_0 = 3
    tau_1 = tau_0/mean_sigma
   
    ax.plot(sojourn_data_range, sojourn_data_pdf, c='#87CEEB', lw=2, ls='-', label='Data')
    ax.plot(sojourn_null_range, sojourn_null_pdf, c='k', lw=2, ls=':', label=r'$\tau \ll \delta t $' + ' (gamma)')

    # #FF6347
    color_tau_all = ['lightskyblue', 'dodgerblue', 'royalblue']
    #tau_all = [1, 2, 3]
    #for tau_idx, tau in enumerate(tau_all):
    #    sojourn_time_range_tau, mixture_dist_tau = make_ou_sojourn_time_dist(tau=tau)
    #    ax.plot(sojourn_time_range_tau, mixture_dist_tau, c=color_tau_all[tau_idx], lw=2, ls='-', label=r'$\tau = $' + str(tau) + ' (OU)')

    sojourn_time_range_tau, mixture_dist_tau = make_ou_sojourn_time_dist(tau_0=0, tau_1=tau_1)
    #ax.plot(sojourn_time_range_tau, mixture_dist_tau, c='dodgerblue', lw=2, ls='-', label='OU, ' + r'$\tau_{i} \propto \sigma_{i}$')

    # 2.715195778605458
    sojourn_time_range_tau, mixture_dist_tau = make_ou_sojourn_time_dist(tau_0=tau_0, tau_1=0)
    ax.plot(sojourn_time_range_tau, mixture_dist_tau, c='k', lw=2, ls='--', label=r'$\tau = 3$')


    ax.set_xlim([1, max(sojourn_data_range)])
    ax.set_ylim([min(sojourn_data_pdf), max([max(sojourn_data_pdf), max(sojourn_null_pdf)])])

    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)



    ax.set_xlabel('Sojourn time (days), ' + r'$\mathcal{T}$', fontsize=12)
    ax.set_ylabel("Probability density, " + r'$P(\mathcal{T})$', fontsize=12)

    ax.legend(loc='upper right', fontsize=10)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_time_data_mix.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



#identify_ml_tau()

plot_sojourn_time_mix_dist()
#make_ou_sojourn_time_dist(max_sojourn_time=100, tau=1)


