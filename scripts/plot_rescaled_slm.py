import os
import random
import copy
import config
import sys
import numpy
import random
import pickle
import scipy.stats as stats
from scipy import integrate
from scipy.spatial import distance
from scipy.special import digamma, gamma


import data_utils
import plot_utils
import stats_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import simulation_utils

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)




slm_dict_path = '%sslm_dict.pickle' % config.data_directory


n_days=1000
n_reps=10000


sigma_range = numpy.logspace(-2, numpy.log10(2), num=10, endpoint=False, base=10)
k_range = numpy.logspace(2, 6, num=10, endpoint=True, base=10)
tau_range = numpy.logspace(numpy.log10(0.1), numpy.log10(100), num=10, endpoint=True, base=10)



def make_rescaled_slm_dict():

    slm_dict = {}

    for sigma in sigma_range:
            
        slm_dict[sigma] = {}
        
        for k in k_range:

            slm_dict[sigma][k] = {}

            for tau in tau_range:

                mean_gamma, cv_gamma = simulation_utils.calculate_mean_and_cv_slm(k, sigma)
                mean_log_gamma = stats_utils.expected_value_log_gamma(mean_gamma, cv_gamma)

                if numpy.isnan(mean_log_gamma) == True:
                    continue

                x_matrix = simulation_utils.simulate_slm_trajectory(n_days=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau, init_log=False, return_log=False)
                x_matrix_log = numpy.log(x_matrix)

                # skip if there are any non finite values after square root transform
                if numpy.isfinite(x_matrix_log).all() == False:
                    continue

                epsilon = 0.1*mean_gamma
                epsilon_log = 0.1*mean_log_gamma

                run_length, mean_run_deviation = simulation_utils.calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                run_length_log, mean_run_deviation_log = simulation_utils.calculate_mean_deviation_pattern_simulation(x_matrix_log, min_run_length=10, min_n_runs=50, epsilon=epsilon_log, x_0=mean_log_gamma)

                if (len(run_length) == 0) or (len(run_length_log) == 0):
                    continue

                print(sigma, k, tau,  max(run_length), max(run_length_log))

                mean_run_deviation_list = [l.tolist() for l in mean_run_deviation]
                mean_run_deviation_log_list = [l.tolist() for l in mean_run_deviation_log]
                
                slm_dict[sigma][k][tau] = {}
                slm_dict[sigma][k][tau]['linear'] = {}
                slm_dict[sigma][k][tau]['linear']['run_length'] = run_length.tolist()
                slm_dict[sigma][k][tau]['linear']['mean_run_deviation'] = mean_run_deviation_list

                slm_dict[sigma][k][tau]['log'] = {}
                slm_dict[sigma][k][tau]['log']['run_length'] = run_length_log.tolist()
                slm_dict[sigma][k][tau]['log']['mean_run_deviation'] = mean_run_deviation_log_list


                slope, intercept, norm_constant = stats_utils.estimate_sojourn_vs_constant_relationship(run_length, mean_run_deviation)
                slope_log, intercept_log, norm_constant_log = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_log, mean_run_deviation_log)

                slm_dict[sigma][k][tau]['linear']['slope'] = slope
                slm_dict[sigma][k][tau]['linear']['intercept'] = intercept
                slm_dict[sigma][k][tau]['linear']['norm_constant'] = norm_constant.tolist()

                slm_dict[sigma][k][tau]['log']['slope'] = slope_log
                slm_dict[sigma][k][tau]['log']['intercept'] = intercept_log
                slm_dict[sigma][k][tau]['log']['norm_constant'] = norm_constant_log.tolist()



    sys.stderr.write("Saving dictionary...\n")
    with open(slm_dict_path, 'wb') as outfile:
        pickle.dump(slm_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



make_rescaled_slm_dict()
