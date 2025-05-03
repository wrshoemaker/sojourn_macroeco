import os
import random
import copy
import config
import sys
import numpy
import math
import random
import pickle
from collections import Counter

import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma, erf
from scipy import interpolate

import stats_utils
import data_utils

import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

numpy.random.seed(123456789)


demog_dict_path = '%sdemog_dict.pickle' % config.data_directory
slm_dict_path = '%sslm_dict.pickle' % config.data_directory


slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sojourn_moment_dict_path = '%ssojourn_moments_dict.pickle' %config.data_directory
sojourn_time_dist_sim_dict_path = '%ssojourn_time_dist_sim_dict.pickle' %config.data_directory

#sojourn_time_dist_sim_fixed_moments_dict_path = '%ssojourn_time_dist_sim_fixed_moments_dict.pickle' %config.data_directory
sojourn_time_dist_sim_fixed_moments_dict_path = '%ssojourn_time_dist_sim_fixed_moments_dict.pickle' % config.data_directory

sojourn_time_dist_sim_fixed_moments_analytic_dict_path = '%ssojourn_time_dist_sim_fixed_moments_analytic_dict.pickle' % config.data_directory


#sigma_range = numpy.logspace(-3, numpy.log10(2), num=10, endpoint=False, base=10)
#k_range = numpy.logspace(2, 6, num=10, endpoint=True, base=10)
tau_range = numpy.logspace(numpy.log10(0.1), numpy.log10(100), num=10, endpoint=True, base=10)

#m_range = numpy.logspace(numpy.log10(1), numpy.log10(10000), num=10, endpoint=True, base=10)
#r_range = numpy.logspace(numpy.log10(0.01), numpy.log10(100), num=10, endpoint=True, base=10)
#D_range = numpy.logspace(numpy.log10(0.01), numpy.log10(100), num=10, endpoint=True, base=10)

# add same timescale
#tau_range = numpy.append(tau_range, 1)
#r_range = numpy.append(r_range, 1)


def calculate_mean_and_cv_slm(k, sigma):

    mean_ = k*(1-(sigma/2))
    cv_ = numpy.sqrt(sigma/(2-sigma))

    return mean_, cv_


def calculate_mean_and_cv_bdm(m, phi):

    #mean_ = m/r
    #cv_ = numpy.sqrt(D/m)

    mean_ = m
    cv_ = phi/numpy.sqrt(2*m)

    return mean_, cv_




def calculate_stationary_params_from_moments(x_bar, cv):

    # calculates parameters for the two models given mean and CV
    # SLM
    sigma = 2*(((cv**-2)+1)**-1)
    k = x_bar*((1 - (sigma/2))**-1)

    # demog
    # timescale inferred from Eq. 3 of Brigatti and Azaele: https://doi.org/10.1038/s41598-024-82882-x
    m = x_bar
    # CV = phi/sqrt(2*m)
    phi = cv*numpy.sqrt(2*m)
    
    return k, sigma, m, phi


def calculate_mean_log_rescaled_gamma(cv):

    beta = cv**(-2)

    return digamma(beta) - numpy.log(beta)




def simulate_slm_trajectory(t_total=None, n_reps=100, k=10000, sigma=1, tau=7, epsilon=0.001, analytic=False, init_log=False, return_log=False):

    # t_total = length of trajectory
    # n_reps = number of replicate trajectories
    # tau = generation timescale of SLM
    # Two timescales: 1) simulation timescale, 2) observation timescale
    # Dimulation timescale = \Delta t = \tau/n_grid_points
    # Observation timesca = daily sampling (set by walk_length)
    # init_log = Whether you are want the initial condition to the expected value of the *log* of x.
    # epsilon = \Delta t / tau ~= 10^{-3}
    # Useful if you are analyzing log transformed empirical data and need simulations

    # delta_t_1 = \Delta t 1 (grid timescale, smaller)
    # delta_t_2 = \Delta t 1 (continuous timescale, larger)

    # \Delta t = numerical integration timestep = length of timeseries / # points
    # \delta t = time between observations (i.e., days, \Delta t << \delta t) 

    # Two types of simulations:
    # 1) Simulations that match data, where we know \delta t (sampling time) and walk_length (length of timeseries). 
    #so \Delta << \delta t << T, where tau can (reasonably) vary
    # 2) Simulations to test analytic predictions, where \Delta t <~ \delta t << \tau << T

    if analytic == False:
        # we need to calcualte # points
        n_points = t_total/(epsilon*tau)
        # we want daily sampling intervals, # points per day
        # make sure we are working with integers
        n_points_per_delta_t_2 = math.ceil(n_points/t_total)
        #delta_t_2 = 1 # *daily* sampling
        # n_points_per_delta_t_2*(t_total/delta_t_2) = n_points_per_delta_t_2*t_total
        n_points_to_integrate = n_points_per_delta_t_2*t_total
        # define delta_t using the actual number of points
        delta_t_1 = t_total/n_points_to_integrate
        # +1 for initial condition
        n_observations = t_total+1

    else:
        # epsilon refers to
        # delta_t = \tau*epsilon
        # T = tau / epsilon
        t_total = round(tau/epsilon)
        delta_t_2 = epsilon*tau
        delta_t_1 = 0.1*delta_t_2
        n_points = t_total/delta_t_1
        # divide number of points by number of delta_t_2 increments
        n_points_per_delta_t_2 = math.ceil(n_points/(t_total/delta_t_2))
        n_points_to_integrate = math.ceil(n_points_per_delta_t_2*(t_total/delta_t_2))
        n_observations = int(n_points_to_integrate/n_points_per_delta_t_2) + 1


    # SLM
    x_bar_slm = k*(1-(sigma/2))
    beta_slm = (2-sigma)/sigma

    # expected value of the log of a gamma random variable.
    # https://stats.stackexchange.com/questions/370880/what-is-the-expected-value-of-the-logarithm-of-gamma-distribution  
    if init_log == True:
        # initial condition is expected value of log of stationary value
        q_0 = digamma(beta_slm) - numpy.log(beta_slm/x_bar_slm) 

    else:
        # initial condition is log of expected stationary value
        q_0 = numpy.log(x_bar_slm)

    
    noise_constant = numpy.sqrt(sigma*delta_t_1/tau)
    
    # +1 for initial condition
    q_matrix = numpy.zeros(shape=(n_observations, n_reps))
    q_matrix[0,:] = q_0

    q_t_minus_one = q_0
    # start with one so we can use "%"
    for t_idx in range(1, n_points_to_integrate+1):
        # we can use t_idx for z because it has one less column than q_matrix
        #q_matrix[t_idx+1,:] = q_t + (1/tau)*(1 - (sigma/2) - numpy.exp(q_t)/k)*delta_t + noise_constant*stats.norm.rvs(size=n_reps)
        q_t = q_t_minus_one + (delta_t_1/tau)*(1 - (sigma/2) - numpy.exp(q_t_minus_one)/k) + noise_constant*stats.norm.rvs(size=n_reps)

        # sampling event
        # supposed to be faster than if statement
        # we want to sample per unit \delta t (delta_t_2)

        match (t_idx % n_points_per_delta_t_2):
            case 0:
                q_matrix[int(t_idx/n_points_per_delta_t_2),:] = q_t

        q_t_minus_one = q_t


    # select samples you want to keep
    #sample_idx = numpy.arange(0, n_iter, n_grid_points)
    #sample_idx = numpy.append(sample_idx, n_iter)
    #q_matrix_sample = q_matrix[sample_idx,:]

    if return_log == True:
        matrix_sample_final = q_matrix

    else:
        matrix_sample_final = numpy.exp(q_matrix)


    return matrix_sample_final




def simulate_bdm_trajectory_dornic(t_total, n_reps, m, phi, tau, delta_t = 1, x_0=None, epsilon=None):

    # Uses the convolution derived by Dornic et al to generate a trajctory of a migration-birth-drift SDE
    # sampling occurs *daily* like in the human gut timeseries
    # https://doi.org/10.1103/PhysRevLett.94.100601

    # delta_t = time between sampling events (1 day) *NOT* gridpoints
    # we are exactly simulating the FPE, not approximating it via Euler
    

    # expected value of the stationary distribution

    if t_total == None:
        delta_t = tau*epsilon
        t_total = tau/epsilon
        n_observations = math.ceil(t_total/delta_t)

    else:
        n_observations = t_total

        
    if x_0 == None:
        #x_0 = m/r
        x_0 = m

    # redefine variables for conveinance
    # first term, constant
    alpha = m/tau
    #beta = -1*r
    beta = -1*(1/tau)
    
    #sigma = numpy.sqrt(2*D)
    sigma = phi/numpy.sqrt(tau)

    x_matrix = numpy.zeros(shape=(n_observations+1, n_reps))
    x_matrix[0,:] = x_0

    lambda_ = (2*beta)/((sigma**2) * (numpy.exp(beta*delta_t)-1) )
    mu_ = -1 + (2*alpha)/(sigma**2)
    
    for i in range(n_observations):
        poisson_rate_i = lambda_*x_matrix[i,:]*numpy.exp(beta*delta_t)
        poisson_rv_i = stats.poisson.rvs(poisson_rate_i)
        gamma_rate_i = mu_ + 1 + poisson_rv_i
        # rescale at each timepoint because this value is used as the initial condition for the next round of sampling.
        #gamma_rv_i = stats.gamma.rvs(gamma_rate_i, size=1)/lambda_
        x_matrix[i+1,:] = (stats.gamma.rvs(gamma_rate_i)/lambda_)
        

    return x_matrix


def simulate_brownian_trajectory(n_days, n_reps, D, x_0, delta_t = 1):

    x_matrix = numpy.zeros(shape=(n_days+1, n_reps))
    x_matrix[0,:] = x_0

    std_dev = numpy.sqrt(2*D*delta_t)

    for i in range(n_days):

        x_matrix[i+1,:] = stats.norm.rvs(loc=x_matrix[i,:], scale=std_dev)

    return x_matrix


#def brownian_semi_infinite_conditional_cdf(x, x_0, D, t):



def simulate_brownian_trajectory_semi_infinite(n_days, n_reps, D, x_0, delta_t = 1):
   
    a = 4*D*delta_t

    def brownian_trajectory_semi_infinite_pdf(x, x_0_vector):
        
        return (a**-1)*numpy.exp(-((x-x_0_vector)**2)/a) - numpy.exp(-((x+x_0_vector)**2)/a)
    
    def brownian_trajectory_semi_infinite_cdf(x, x_0_vector):
        
        return 0.5*(erf((x-x_0_vector)/numpy.sqrt(a)) - erf((x+x_0_vector)/numpy.sqrt(a)) + 2*erf(x_0_vector/numpy.sqrt(a)))




    #def sample_pdf(pdf_, x_0_vector, max_x=10000):

    #    x = numpy.linspace(0, max_x, 1000000000)
    #    #pdf_x = pfd_(x, x_0_vector, a)
    #    #cdf_x = numpy.cumsum(pdf_x)
    #    cdf_x = cdf_x/cdf_x.max()
    #    inverse_cdf = interpolate.interp1d(cdf_x, x)
        
    #    return inverse_cdf
    
    def sample_cdf(cdf_, x_0_vector, max_x=10000):

        x = numpy.linspace(0, max_x, 10000000)
        #pdf_x = pfd_(x, x_0_vector, a)
        cdf_x = cdf_(x, x_0_vector)
        cdf_x = cdf_x/cdf_x.max()
        inverse_cdf = interpolate.interp1d(cdf_x, x)
        
        return inverse_cdf

   
    x_matrix = numpy.zeros(shape=(n_days+1, n_reps))
    x_matrix[0,:] = x_0

    uniform_rv = stats.uniform.rvs(size=(n_days, n_reps))

    for i in range(n_days):

        x_matrix[i+1,:] = sample_cdf(brownian_trajectory_semi_infinite_pdf, x_matrix[i,:])(uniform_rv[i,:], x_matrix[i,:])

        
        #stats.norm.rvs(loc=x_matrix[i,:], scale=std_dev)
    return x_matrix




def fit_slope_simulation(x_matrix, min_run_length=10, min_n_obs=10):

    deviation_x = x_matrix[1:,:] - x_matrix[0,:]

    slope_all = []

    for deviation_traj in deviation_x.T:

        run_values, run_starts, run_lengths = data_utils.find_runs(deviation_traj>0)
        run_sum = numpy.asarray([sum(numpy.absolute(deviation_traj[run_starts[run_j_idx]:run_starts[run_j_idx]+run_lengths[run_j_idx]])) for run_j_idx in range(len(run_lengths))])
        
        to_keep_idx = (run_lengths >= min_run_length) & (run_sum>0)
        run_length_filter = run_lengths[to_keep_idx]
        run_sum_filter = run_sum[to_keep_idx]

        # only fit regressions if there are at least 10 samples.
        if len(run_sum_filter) < min_n_obs:
            continue

        slope, intercept, r_valuer_value, p_value, std_err = data_utils.stats.linregress(numpy.log10(run_length_filter), numpy.log10(run_sum_filter))
        
        if numpy.isnan(slope) == True:
            continue


        slope_all.append(slope)

    return slope_all



def calculate_moments_sojourn_time(x_matrix):

    deviation_x = x_matrix[1:,:] - x_matrix[0,:]

    mean_run_length_all = []
    std_run_length_all = []
    max_run_length_all = []
    for deviation_traj in deviation_x.T:

        run_values, run_starts, run_lengths = data_utils.find_runs(deviation_traj>0)
        mean_run_length_all.append(numpy.mean(run_lengths))
        std_run_length_all.append(numpy.std(run_lengths))
        max_run_length_all.append(max(run_lengths))

    return mean_run_length_all, std_run_length_all, max_run_length_all



def calculate_sojourn_time_dist(x_matrix, min_run_length=1, x_0=None):

    if x_0 != None:
        deviation_x = x_matrix[1:,:] - x_0

    else:
        deviation_x = x_matrix[1:,:] - x_matrix[0,:]

    run_lengths_all = []

    for deviation_traj in deviation_x.T:

        run_values, run_starts, run_lengths = data_utils.find_runs(deviation_traj>0, min_run_length=min_run_length)
        run_lengths_all.append(run_lengths)

    run_lengths_all = numpy.concatenate(run_lengths_all).ravel()

    return run_lengths_all

    



def calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=10, epsilon=None, x_0=None):

    # min_n_runs = minimum number of replicate trajectories belonging to a given sojourn time to be included in the analysis
    # x_0 = initial condition. Default is that none is provided. If it is provided, it is used. 
    # relative deviation from origin: epsilon = (x(t) - x(0))/x(0)
    # Absolute value of the relative deviation must be within +/- epsilon at start and end of sojourn period
    # This code return the mean deviation from the origin over a large number of replicate sojourns
    # It does *not* rescale the mean deviation

    if x_0 != None:
        deviation_x = x_matrix[1:,:] - x_0

    else:
        deviation_x = x_matrix[1:,:] - x_matrix[0,:]
    

    run_dict = {}

    for deviation_traj in deviation_x.T:

        run_values, run_starts, run_lengths = data_utils.find_runs(deviation_traj>0, min_run_length=min_run_length)

        #run_lengths_all.append(run_lengths)
        for run_j_idx in range(len(run_values)):
            
            run_values_j = run_values[run_j_idx]
            run_starts_j = run_starts[run_j_idx]
            run_length_j = run_lengths[run_j_idx]

            # cannot have a sojourn time that is the length of the entire timeseries
            # weird fluctuations here, ignore..
            if (run_length_j >= deviation_x.shape[0]-3):
                continue


            run_deviation_j = data_utils.extract_trajectory_epsilon(deviation_traj, run_values_j, run_starts_j, run_length_j, epsilon=epsilon)

            if run_deviation_j is None:
                continue

            #if epsilon != None:
                
            #    # avoid issue of walk starting at zero
            #    # inclusive
            #    start_before = abs(deviation_traj[max([(run_starts_j-1), 0])])
            #    start_after = abs(deviation_traj[run_starts_j])
            #    # inclusive
            #    end_before = abs(deviation_traj[(run_starts_j + run_length_j-1)])
            #    end_after = abs(deviation_traj[min([(len(deviation_traj)-1), (run_starts_j + run_length_j)])])

            #    start_before_bool = False
            #    start_after_bool = False
            #    end_before_bool = False
            #    end_after_bool = False
                
            #    if start_before <= epsilon:
            #        start_before_bool = True 

            #    if start_after <= epsilon:
            #        start_after_bool = True 

            #    if end_before <= epsilon:
            #        end_before_bool = True 

            #    if end_after <= epsilon:
            #        end_after_bool = True 

            #    # continue, not within epsilon at either option
            #    if ((start_before_bool+start_after_bool) == 0) or ((end_before_bool+end_after_bool) == 0):
            #        continue
                    
            #    # Baldassarri used the first timepoint that reached epsilon 
            #    # inclusive
            #    if start_before_bool == True:
            #        new_run_start_j = max([(run_starts_j-1), 0])
            #    else:
            #        new_run_start_j = run_starts_j
                
            #    # first timepoint at end that reached epsilon
            #    # exclusive
            #    if end_before == True:
            #        new_run_end_j = run_starts_j + run_length_j
            #    else:
            #        new_run_end_j = run_starts_j + run_length_j + 1

            #    if new_run_start_j == new_run_end_j:
            #        continue

            #    run_deviation_j = numpy.absolute(deviation_traj[new_run_start_j:new_run_end_j])

            #else:
            #    run_deviation_j = numpy.absolute(deviation_traj[run_starts_j:(run_starts_j + run_length_j)])

            # have to check again...
            #if len(run_deviation_j) == deviation_x.shape[0]:
            #    continue

            
            run_lengths_new_j = len(run_deviation_j)
            if run_lengths_new_j not in run_dict:
                run_dict[run_lengths_new_j] = []

            #run_lengths_all.append(len(run_deviation_j))
            #run_sum_all.append(sum(run_deviation_j))
            #run_deviation_all.append(run_deviation_j)

            run_dict[run_lengths_new_j].append(run_deviation_j)


    #run_lengths_all = numpy.concatenate(run_lengths_all).ravel()
    #run_lengths_all = numpy.asarray(run_lengths_all)
    #run_sum_all = numpy.asarray(run_sum_all, dtype=object)
    #run_deviation_all = numpy.asarray(run_deviation_all, dtype=object)

    # sorts the set
    #run_lengths_set = numpy.asarray(list(set(run_lengths_all.tolist())))
    #run_lengths_set = numpy.unique(run_lengths_all)
    run_lengths_set = list(run_dict.keys())
    run_lengths_set.sort()

    run_length_all_final = []
    mean_run_deviation_all_final = []

    #for run_length_i in run_lengths_set:
    for run_length_i in run_lengths_set:

        run_deviation_i = run_dict[run_length_i]

        #idx_i = (run_lengths_all==run_length_i)

        # Insufficient number of replicates
        #if sum(idx_i) < min_n_runs:
        #    continue

        if len(run_deviation_i) < min_n_runs:
            continue

        run_length_all_final.append(run_length_i)
        mean_run_deviation_all_final.append(numpy.mean(run_deviation_i, axis=0))
        
        # slice and form into matrix.
        #run_deviation_matrix_i = numpy.vstack(run_deviation_all[idx_i])
        # try rescaling by sum within each replicate
        #run_deviation_matrix_i = run_deviation_matrix_i/numpy.sum(run_deviation_matrix_i, axis=1)
        #run_deviation_matrix_i = ((run_deviation_matrix_i.T)/numpy.sum(run_deviation_matrix_i, axis=1)).T
        #rescaled_mean_run_deviation_i = mean_run_deviation_i#/sum(mean_run_deviation_i)

        #mean_run_deviation_i = numpy.mean(run_deviation_matrix_i, axis=0)
        #mean_run_deviation_i = numpy.mean(run_deviation_all[idx_i])
        #run_length_all_final.append(run_length_i)
        #rescaled_mean_run_deviation_all.append(mean_run_deviation_i)

    # should already be sorted
    run_length_all_final = numpy.asarray(run_length_all_final)
    mean_run_deviation_all_final = numpy.asarray(mean_run_deviation_all_final, dtype=object)


    return run_length_all_final, mean_run_deviation_all_final
    




def simulate_cv_sojourn_time(n_days, n_reps):

    sojourn_dict = {}
    sojourn_dict['slm'] = {}
    sojourn_dict['demog'] = {}

    print('Running SLM....')

    for sigma in sigma_range:
        
        sojourn_dict['slm'][sigma] = {}
        
        for k in k_range:

            sojourn_dict['slm'][sigma][k] = {}

            for tau in tau_range:

                x_matrix = simulate_slm_trajectory(n_days=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau)
                mean_run_length_all, std_run_length_all, max_run_length_all = calculate_moments_sojourn_time(x_matrix)

                sojourn_dict['slm'][sigma][k][tau] = {}
                sojourn_dict['slm'][sigma][k][tau]['mean_sojourn_time'] = mean_run_length_all
                sojourn_dict['slm'][sigma][k][tau]['std_sojourn_time'] = std_run_length_all
                sojourn_dict['slm'][sigma][k][tau]['max_sojourn_time'] = max_run_length_all

                mean_cv = numpy.mean(numpy.asarray(std_run_length_all)/numpy.asarray(mean_run_length_all))
                mean_max = numpy.mean(max_run_length_all)

                x_bar_slm = k*(1-(sigma/2))
                #print((2-sigma)/sigma)
                cv_slm = numpy.sqrt(sigma/(2-sigma))

                print(round(cv_slm, 3), round(x_bar_slm, 3), round(tau, 3), round(mean_cv, 3), round(mean_max, 3))



    print('Running demog....')
    for m in m_range:
        
        sojourn_dict['demog'][m] = {}
        
        for r in r_range:

            sojourn_dict['demog'][m][r] = {}

            for D in D_range:

                x_matrix = simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D)
                mean_run_length_all, std_run_length_all, max_run_length_all = calculate_moments_sojourn_time(x_matrix)

                sojourn_dict['demog'][m][r][D] = {}
                sojourn_dict['demog'][m][r][D]['mean_sojourn_time'] = mean_run_length_all
                sojourn_dict['demog'][m][r][D]['std_sojourn_time'] = std_run_length_all
                sojourn_dict['demog'][m][r][D]['max_sojourn_time'] = max_run_length_all

                mean_cv = numpy.mean(numpy.asarray(std_run_length_all)/numpy.asarray(mean_run_length_all))
                mean_max = numpy.mean(max_run_length_all)
                x_bar_demog = m/r
                cv_demog = numpy.sqrt(D/m)

                # 1/r = autocorrelation timescale
                print(round(cv_demog, 3), round(x_bar_demog, 3), round(1/r, 3), round(mean_cv, 3), round(mean_max, 3))


    sys.stderr.write("Saving dictionary...\n")
    with open(sojourn_moment_dict_path, 'wb') as outfile:
        pickle.dump(sojourn_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def simulate_sojourn_time_vs_cumulative_walk_length_exponent(n_days, n_reps, n_slopes=100, min_run_length=10):

    slope_dict = {}
    slope_dict['slm'] = {}
    slope_dict['demog'] = {}

    print('Running SLM....')

    for sigma in sigma_range:
        
        slope_dict['slm'][sigma] = {}
        
        for k in k_range:

            slope_dict['slm'][sigma][k] = {}

            for tau in tau_range:

                slope_dict['slm'][sigma][k][tau] = {}
                slope_dict['slm'][sigma][k][tau]['slope'] = []
                
                while len(slope_dict['slm'][sigma][k][tau]['slope']) < n_slopes:
                    
                    x_matrix = simulate_slm_trajectory(n_days=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau)
                    slope_all = fit_slope_simulation(x_matrix, min_run_length=min_run_length)
                    n_slopes_to_add = n_slopes - len(slope_dict['slm'][sigma][k][tau]['slope'])

                    if len(slope_all) > 0:
                        slope_dict['slm'][sigma][k][tau]['slope'].extend(slope_all[0:min([len(slope_all), n_slopes_to_add])])


                #print(sigma, k, tau, numpy.mean(slope_dict['slm'][sigma][k][tau]['slope']))
                x_bar_slm = k*(1-(sigma/2))
                cv_slm = numpy.sqrt(sigma/(2-sigma))
                mean_slope = numpy.mean(slope_dict['slm'][sigma][k][tau]['slope'])
                se_slope = numpy.std(slope_dict['slm'][sigma][k][tau]['slope'])/numpy.sqrt(n_slopes)
                print(round(cv_slm, 3), round(x_bar_slm, 3), round(tau, 3), round(mean_slope, 3), round(se_slope, 3))


    print('Running demog....')

    for m in m_range:
        
        slope_dict['demog'][m] = {}
        
        for r in r_range:

            slope_dict['demog'][m][r] = {}

            for D in D_range:

                slope_dict['demog'][m][r][D] = {}
                slope_dict['demog'][m][r][D]['slope'] = []

                while len(slope_dict['demog'][m][r][D]['slope']) < n_slopes:
                    
                    x_matrix = simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D)
                    slope_all = fit_slope_simulation(x_matrix, min_run_length=min_run_length)
                    n_slopes_to_add = n_slopes - len(slope_dict['demog'][m][r][D]['slope'])

                    if len(slope_all) > 0:
                        slope_dict['demog'][m][r][D]['slope'].extend(slope_all[0:min([len(slope_all), n_slopes_to_add])])
                
                
                x_bar_demog = m/r
                cv_demog = numpy.sqrt(D/m)
                #print(m, r, D, numpy.mean(slope_dict['demog'][m][r][D]['slope']) )
                mean_slope = numpy.mean(slope_dict['demog'][m][r][D]['slope'])
                se_slope = numpy.std(slope_dict['demog'][m][r][D]['slope'])/numpy.sqrt(n_slopes)
                print(round(cv_demog, 3), round(x_bar_demog, 3), round(1/r, 3), round(mean_slope, 3), round(se_slope, 3))



    sys.stderr.write("Saving slope dictionary...\n")
    with open(slope_dict_path, 'wb') as outfile:
        pickle.dump(slope_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")




def simulate_sojourn_time_dist():

    # We want to understand how the sojourn time distribution varies across parameter regimes and models
    # We are NOT using the epsilon criteria. We are using all sojourns. 

    n_days=1000
    n_reps=10000

    dist_dict = {}
    dist_dict['demog'] = {}
    dist_dict['slm'] = {}

    dist_dict['n_days'] = n_days
    dist_dict['n_reps'] = n_reps

    # First, demog
    for m in m_range:
            
        dist_dict['demog'][m] = {}
        
        for r in r_range:

            dist_dict['demog'][m][r] = {}

            for D in D_range:

                mean_gamma, cv_gamma = calculate_mean_and_cv_demog(m, r, D)
                mean_square_root_gamma = stats_utils.expected_value_square_root_gamma(mean_gamma, cv_gamma)

                # unlikely that the square root transfomation will yield useful results
                if numpy.isnan(mean_square_root_gamma) == True:
                    continue

                # initial condition is expected stationary value of square root of gamma rv
                x_matrix = simulate_bdm_trajectory_dornic(n_days, n_reps, m, r, D, x_0=mean_gamma)
                #x_matrix_sqrt = numpy.sqrt(x_matrix)
                
                # skip if there are any non finite values after square root transform
                #if numpy.isfinite(x_matrix_sqrt).all() == False:
                #    continue

                if numpy.isfinite(x_matrix).all() == False:
                    continue

                run_lengths = calculate_sojourn_time_dist(x_matrix, min_run_length=1)
                #run_lengths_sqrt = calculate_sojourn_time_dist(x_matrix_sqrt, min_run_length=1, x_0=mean_square_root_gamma)

                print('Demog', m, r, D)

                dist_dict['demog'][m][r][D] = {}
                dist_dict['demog'][m][r][D]['linear'] = {}
                dist_dict['demog'][m][r][D]['linear']['sojourn_time_count_dict'] = dict(Counter(run_lengths.tolist()))
                dist_dict['demog'][m][r][D]['linear']['mean_sojourn_time'] = numpy.mean(run_lengths)
                dist_dict['demog'][m][r][D]['linear']['std_sojourn_time'] = numpy.std(run_lengths)


                
    # then, SLM
    for sigma in sigma_range:
            
        dist_dict['slm'][sigma] = {}
        
        for k in k_range:

            dist_dict['slm'][sigma][k] = {}

            for tau in tau_range:

                mean_gamma, cv_gamma = calculate_mean_and_cv_slm(k, sigma)
                mean_log_gamma = stats_utils.expected_value_log_gamma(mean_gamma, cv_gamma)

                if numpy.isnan(mean_log_gamma) == True:
                    continue
                
                x_matrix = simulate_slm_trajectory(n_days=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau, init_log=False, return_log=False)
                #x_matrix_log = numpy.log(x_matrix)

                #if numpy.isfinite(x_matrix_log).all() == False:
                #    continue

                if numpy.isfinite(x_matrix).all() == False:
                    continue

                run_lengths = calculate_sojourn_time_dist(x_matrix, min_run_length=1)
                #run_lengths_log = calculate_sojourn_time_dist(x_matrix_log, min_run_length=1, x_0=mean_log_gamma)

                print('SLM', sigma, k, tau)

                dist_dict['slm'][sigma][k][tau] = {}
                #dist_dict['slm'][sigma][k][tau]['log'] = dict(Counter(run_lengths_log.tolist()))
                dist_dict['slm'][sigma][k][tau]['linear'] = {}
                dist_dict['slm'][sigma][k][tau]['linear']['sojourn_time_count_dict'] = dict(Counter(run_lengths.tolist()))
                dist_dict['slm'][sigma][k][tau]['linear']['mean_sojourn_time'] = numpy.mean(run_lengths)
                dist_dict['slm'][sigma][k][tau]['linear']['std_sojourn_time'] = numpy.std(run_lengths)


    sys.stderr.write("Saving dictionary...\n")
    with open(sojourn_time_dist_sim_dict_path, 'wb') as outfile:
        pickle.dump(dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")

    # mylist = [key for key, val in mydict.items() for _ in range(val)]
    





def make_demog_dict(n_days=1000, n_reps=10000):

    demog_dict = {}

    for m in m_range:
            
        demog_dict[m] = {}
        
        for r in r_range:

            demog_dict[m][r] = {}

            for D in D_range:

                mean_gamma, cv_gamma = calculate_mean_and_cv_demog(m, r, D)
                mean_square_root_gamma = stats_utils.expected_value_square_root_gamma(mean_gamma, cv_gamma)

                # unlikely that the square root transfomation will yield useful results
                if numpy.isnan(mean_square_root_gamma) == True:
                    continue

                # initial condition is expected stationary value of square root of gamma rv
                x_matrix = simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, x_0=mean_gamma)
                # First timepoint has already been transformed....
                x_matrix_sqrt = numpy.sqrt(x_matrix)
                
                # skip if there are any non finite values after square root transform
                if numpy.isfinite(x_matrix_sqrt).all() == False:
                    continue

                epsilon = 0.1*mean_gamma
                epsilon_sqrt = 0.1*mean_square_root_gamma

                run_length, mean_run_deviation = calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                # epsilon for square root and use expected value of square root stationary gamma rv as initial condition...
                run_length_sqrt, mean_run_deviation_sqrt = calculate_mean_deviation_pattern_simulation(x_matrix_sqrt, min_run_length=10, min_n_runs=50, epsilon=epsilon_sqrt, x_0=mean_square_root_gamma)

                if (len(run_length) == 0) or (len(run_length_sqrt) == 0):
                    continue

                print(m, r, D,  max(run_length), max(run_length_sqrt))

                mean_run_deviation_list = [l.tolist() for l in mean_run_deviation]
                mean_run_deviation_sqrt_list = [l.tolist() for l in mean_run_deviation_sqrt]

                demog_dict[m][r][D] = {}
                demog_dict[m][r][D]['linear'] = {}
                demog_dict[m][r][D]['linear']['run_length'] = run_length.tolist()
                demog_dict[m][r][D]['linear']['mean_run_deviation'] = mean_run_deviation_list

                demog_dict[m][r][D]['sqrt'] = {}
                demog_dict[m][r][D]['sqrt']['run_length'] = run_length_sqrt.tolist()
                demog_dict[m][r][D]['sqrt']['mean_run_deviation'] = mean_run_deviation_sqrt_list

                # calculate slope for sojourn time vs. norm. constant

                for data_type in ['linear', 'sqrt']:

                    if data_type == 'linear':
                        run_length_ = run_length
                        mean_run_deviation_ = mean_run_deviation

                    else:
                        run_length_ = run_length_sqrt
                        mean_run_deviation_ = mean_run_deviation_sqrt

                    slope, intercept, norm_constant = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_, mean_run_deviation_)

                    demog_dict[m][r][D][data_type]['slope'] = slope
                    demog_dict[m][r][D][data_type]['intercept'] = intercept
                    demog_dict[m][r][D][data_type]['norm_constant'] = norm_constant.tolist()



    sys.stderr.write("Saving dictionary...\n")
    with open(demog_dict_path, 'wb') as outfile:
        pickle.dump(demog_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")






def make_slm_dict(n_days=1000, n_reps=10000):

    slm_dict = {}

    for sigma in sigma_range:
            
        slm_dict[sigma] = {}
        
        for k in k_range:

            slm_dict[sigma][k] = {}

            for tau in tau_range:

                mean_gamma, cv_gamma = calculate_mean_and_cv_slm(k, sigma)
                mean_log_gamma = stats_utils.expected_value_log_gamma(mean_gamma, cv_gamma)

                if numpy.isnan(mean_log_gamma) == True:
                    continue

                x_matrix = simulate_slm_trajectory(n_days=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau, init_log=False, return_log=False)
                x_matrix_log = numpy.log(x_matrix)

                # skip if there are any non finite values after square root transform
                if numpy.isfinite(x_matrix_log).all() == False:
                    continue

                epsilon = 0.1*mean_gamma
                epsilon_log = 0.1*mean_log_gamma

                run_length, mean_run_deviation = calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                run_length_log, mean_run_deviation_log = calculate_mean_deviation_pattern_simulation(x_matrix_log, min_run_length=10, min_n_runs=50, epsilon=epsilon_log, x_0=mean_log_gamma)

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



def simulate_sojourn_time_dist_fixed_mean_cv(n_days=1000, n_reps=10000):

    mean_range = [10000]
    cv_range = numpy.logspace(numpy.log10(0.01), numpy.log10(10), num=10, endpoint=True, base=10)
    #cv_range = [10]

    dist_dict = {}

    for mean in mean_range:

        dist_dict[mean] = {}

        for cv in cv_range:

            dist_dict[mean][cv] = {}

            k, sigma, m, phi = calculate_stationary_params_from_moments(mean, cv)

            for tau in tau_range:

                print(mean, cv, tau)

                x_matrix_bdm = simulate_bdm_trajectory_dornic(t_total=n_days, n_reps=n_reps, m=m, phi=phi, tau=tau)
                x_matrix_slm = simulate_slm_trajectory(t_total=n_days, n_reps=n_reps, k=k, sigma=sigma, tau=tau, epsilon=0.001, analytic=False)

                if (numpy.isfinite(x_matrix_bdm).all() == False) or (numpy.isfinite(x_matrix_slm).all() == False):
                    continue

                run_lengths_bdm = calculate_sojourn_time_dist(x_matrix_bdm, min_run_length=1)
                run_lengths_slm = calculate_sojourn_time_dist(x_matrix_slm, min_run_length=1)

                # get the sojourn deviation
                epsilon = 0.05*mean

                run_length_deviation_all_bdm, mean_run_deviation_bdm = calculate_mean_deviation_pattern_simulation(x_matrix_bdm, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                run_length_deviation_all_slm, mean_run_deviation_slm = calculate_mean_deviation_pattern_simulation(x_matrix_slm, min_run_length=10, min_n_runs=50, epsilon=epsilon)

                mean_run_deviation_bdm_list = [l.tolist() for l in mean_run_deviation_bdm]
                mean_run_deviation_slm_list = [l.tolist() for l in mean_run_deviation_slm]

                slope_bdm, intercept_bdm, norm_constant_bdm = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_deviation_all_bdm, mean_run_deviation_bdm)
                slope_slm, intercept_slm, norm_constant_slm = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_deviation_all_slm, mean_run_deviation_slm)

                print(slope_bdm, len(norm_constant_bdm), slope_slm, len(norm_constant_slm))

                tau_float = float(tau)
                dist_dict[mean][cv][tau_float] = {}
                
                dist_dict[mean][cv][tau_float]['bdm'] = {}
                dist_dict[mean][cv][tau_float]['bdm']['sojourn_time_count_dict'] = dict(Counter(run_lengths_bdm.tolist()))
                dist_dict[mean][cv][tau_float]['bdm']['mean_sojourn_time'] = numpy.mean(run_lengths_bdm)
                #dist_dict[mean][cv][tau_float]['bdm']['std_sojourn_time'] = numpy.std(run_lengths_bdm)
                
                dist_dict[mean][cv][tau_float]['bdm']['run_length'] = run_length_deviation_all_bdm.tolist()
                dist_dict[mean][cv][tau_float]['bdm']['mean_run_deviation'] = mean_run_deviation_bdm_list
                dist_dict[mean][cv][tau_float]['bdm']['slope'] = slope_bdm
                dist_dict[mean][cv][tau_float]['bdm']['intercept'] = intercept_bdm
                dist_dict[mean][cv][tau_float]['bdm']['norm_constant'] = norm_constant_bdm.tolist()


                dist_dict[mean][cv][tau_float]['slm'] = {}
                dist_dict[mean][cv][tau_float]['slm']['sojourn_time_count_dict'] = dict(Counter(run_lengths_slm.tolist()))
                dist_dict[mean][cv][tau_float]['slm']['mean_sojourn_time'] = numpy.mean(run_lengths_slm)
                #dist_dict[mean][cv][tau_float]['slm']['std_sojourn_time'] = numpy.std(run_lengths_slm)
               
                dist_dict[mean][cv][tau_float]['slm']['run_length'] = run_length_deviation_all_slm.tolist()
                dist_dict[mean][cv][tau_float]['slm']['mean_run_deviation'] = mean_run_deviation_slm_list
                dist_dict[mean][cv][tau_float]['slm']['slope'] = slope_slm
                dist_dict[mean][cv][tau_float]['slm']['intercept'] = intercept_slm
                dist_dict[mean][cv][tau_float]['slm']['norm_constant'] = norm_constant_slm.tolist()


    
    sys.stderr.write("Saving dictionary...\n")
    with open(sojourn_time_dist_sim_fixed_moments_dict_path, 'wb') as outfile:
        pickle.dump(dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def simulate_sojourn_time_dist_fixed_mean_cv_analytic(t_total=None, n_reps=10000):

    mean_range = [10000]
    cv_range = numpy.logspace(numpy.log10(0.01), numpy.log10(10), num=10, endpoint=True, base=10)
    #cv_range = [10]

    dist_dict = {}

    for mean in mean_range:

        dist_dict[mean] = {}

        for cv in cv_range:

            dist_dict[mean][cv] = {}

            k, sigma, m, phi = calculate_stationary_params_from_moments(mean, cv)

            for tau in tau_range:

                print(mean, cv, tau)

                epsilon_sim = 0.01

                x_matrix_bdm = simulate_bdm_trajectory_dornic(t_total=t_total, n_reps=n_reps, m=m, phi=phi, tau=tau, epsilon=epsilon_sim)
                x_matrix_slm = simulate_slm_trajectory(t_total=t_total, n_reps=n_reps, k=k, sigma=sigma, tau=tau, epsilon=epsilon_sim, analytic=True)

                if (numpy.isfinite(x_matrix_bdm).all() == False) or (numpy.isfinite(x_matrix_slm).all() == False):
                    continue

                run_lengths_bdm = calculate_sojourn_time_dist(x_matrix_bdm, min_run_length=1)
                run_lengths_slm = calculate_sojourn_time_dist(x_matrix_slm, min_run_length=1)

                # get the sojourn deviation
                epsilon = 0.05*mean

                run_length_deviation_all_bdm, mean_run_deviation_bdm = calculate_mean_deviation_pattern_simulation(x_matrix_bdm, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                run_length_deviation_all_slm, mean_run_deviation_slm = calculate_mean_deviation_pattern_simulation(x_matrix_slm, min_run_length=10, min_n_runs=50, epsilon=epsilon)

                mean_run_deviation_bdm_list = [l.tolist() for l in mean_run_deviation_bdm]
                mean_run_deviation_slm_list = [l.tolist() for l in mean_run_deviation_slm]

                slope_bdm, intercept_bdm, norm_constant_bdm = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_deviation_all_bdm, mean_run_deviation_bdm)
                slope_slm, intercept_slm, norm_constant_slm = stats_utils.estimate_sojourn_vs_constant_relationship(run_length_deviation_all_slm, mean_run_deviation_slm)

                #print(slope_bdm, len(norm_constant_bdm), slope_slm, len(norm_constant_slm))

                tau_float = float(tau)
                dist_dict[mean][cv][tau_float] = {}
                
                dist_dict[mean][cv][tau_float]['bdm'] = {}
                dist_dict[mean][cv][tau_float]['bdm']['sojourn_time_count_dict'] = dict(Counter(run_lengths_bdm.tolist()))
                dist_dict[mean][cv][tau_float]['bdm']['mean_sojourn_time'] = numpy.mean(run_lengths_bdm)
                #dist_dict[mean][cv][tau_float]['bdm']['std_sojourn_time'] = numpy.std(run_lengths_bdm)
                
                dist_dict[mean][cv][tau_float]['bdm']['run_length'] = run_length_deviation_all_bdm.tolist()
                dist_dict[mean][cv][tau_float]['bdm']['mean_run_deviation'] = mean_run_deviation_bdm_list
                dist_dict[mean][cv][tau_float]['bdm']['slope'] = slope_bdm
                dist_dict[mean][cv][tau_float]['bdm']['intercept'] = intercept_bdm
                dist_dict[mean][cv][tau_float]['bdm']['norm_constant'] = norm_constant_bdm.tolist()


                dist_dict[mean][cv][tau_float]['slm'] = {}
                dist_dict[mean][cv][tau_float]['slm']['sojourn_time_count_dict'] = dict(Counter(run_lengths_slm.tolist()))
                dist_dict[mean][cv][tau_float]['slm']['mean_sojourn_time'] = numpy.mean(run_lengths_slm)
                #dist_dict[mean][cv][tau_float]['slm']['std_sojourn_time'] = numpy.std(run_lengths_slm)
               
                dist_dict[mean][cv][tau_float]['slm']['run_length'] = run_length_deviation_all_slm.tolist()
                dist_dict[mean][cv][tau_float]['slm']['mean_run_deviation'] = mean_run_deviation_slm_list
                dist_dict[mean][cv][tau_float]['slm']['slope'] = slope_slm
                dist_dict[mean][cv][tau_float]['slm']['intercept'] = intercept_slm
                dist_dict[mean][cv][tau_float]['slm']['norm_constant'] = norm_constant_slm.tolist()


    
    sys.stderr.write("Saving dictionary...\n")
    with open(sojourn_time_dist_sim_fixed_moments_analytic_dict_path, 'wb') as outfile:
        pickle.dump(dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")






if __name__ == "__main__":

    print("Running...")

    #make_demog_dict()
    # make_slm_dict()

    #simulate_sojourn_time_dist()

    #simulate_sojourn_time_dist_fixed_mean_cv()


    simulate_sojourn_time_dist_fixed_mean_cv_analytic()

  



#x_matrix = simulate_demog_trajectory_dornic(n_days, 10000, 100, 0.1, 1)

#run_lengths_set, rescaled_mean_run_deviation_all = calculate_mean_deviation_pattern_simulation(x_matrix)



#simulate_sojourn_time_vs_cumulative_walk_length_exponent(n_days, n_reps, n_slopes=10)


#simulate_cv_sojourn_time(n_days, n_reps)

