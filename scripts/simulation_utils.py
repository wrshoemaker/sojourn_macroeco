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
from scipy.special import digamma, gamma, erf
from scipy import interpolate

import stats_utils
import data_utils


numpy.random.seed(123456789)

slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sojourn_moment_dict_path = '%ssojourn_moments_dict.pickle' %config.data_directory


sojourn_time_dist_sim_dict_path = '%ssojourn_time_dist_sim_dict.pickle' %config.data_directory


def calculate_mean_and_cv_slm(k, sigma):

    mean_ = k*(1-(sigma/2))
    cv_ = numpy.sqrt(sigma/(2-sigma))

    return mean_, cv_


def calculate_mean_and_cv_demog(m, r, D):

    mean_ = m/r
    cv_ = numpy.sqrt(D/m)

    return mean_, cv_



def calculate_params_for_slm_and_demog(x_bar, cv, timescale):

    # calculates parameters for the two models given mean, CV, and timescale

    # SLM
    sigma = 2*(((cv**-2)+1)**-1)
    k = x_bar*((1 - (sigma/2))**-1)
    tau = timescale

    # demog
    # timescale inferred from Eq. 3 of Brigatti and Azaele: https://doi.org/10.1038/s41598-024-82882-x
    r = 1/timescale
    m = r*x_bar
    D = (cv**2)*m
    
    # So we don't have to keep track of the order of returned variables
    param_dict = {}
    param_dict['k'] = k
    param_dict['sigma'] = sigma
    param_dict['tau'] = tau
    
    param_dict['m'] = m
    param_dict['D'] = D
    param_dict['r'] = r

    return param_dict





def simulate_slm_trajectory(n_days=1000, n_reps=100, k=10000, sigma=1, tau=7, n_grid_points=500, init_log=False, return_log=False):

    # n_days = length of trajectory
    # n_reps = number of replicate trajectories
    # tau = generation timescale of SLM
    # Two timescales: 1) simulation timescale, 2) observation timescale
    # Dimulation timescale = \Delta t = \tau/n_grid_points
    # Observation timesca = daily sampling (set by n_days)
    # init_log = Whether you are want the initial condition to the expected value of the *log* of x. 
    # Useful if you are analyzing log transformed empirical data and need simulations

    # SLM
    x_bar_slm = k*(1- (sigma/2))
    beta_slm = (2-sigma)/sigma

    # expected value of the log of a gamma random variable.
    # https://stats.stackexchange.com/questions/370880/what-is-the-expected-value-of-the-logarithm-of-gamma-distribution  
    if init_log == True:
        # initial condition is expected value of log of stationary value
        q_0 = digamma(beta_slm) - numpy.log(beta_slm/x_bar_slm) 

    else:
        # initial condition is log of expected stationary value
        q_0 = numpy.log(x_bar_slm)

    # total number of iterations
    n_iter = n_grid_points*n_days
    delta_t = tau/n_grid_points
    noise_constant = numpy.sqrt(sigma*delta_t/tau)
    
    # +1 for initial condition
    #q_matrix = numpy.zeros(shape=(n_iter+1, n_reps))

    q_matrix = numpy.zeros(shape=(n_days+1, n_reps))
    q_matrix[0,:] = q_0

    #z = numpy.random.randn(n_iter, n_reps)
    #z = stats.norm.rvs(size=(n_iter, n_reps))
    # Multiply z by its constants
    #z = z*numpy.sqrt(sigma*delta_t/tau)
    
    q_t_minus_one = q_0
    # add one so we can use "%"
    for t_idx in range(1, n_iter+1):
        #for t_idx in range(n_iter):

        #_t = q_matrix[t_idx,:]
        # we can use t_idx for z because it has one less column than q_matrix
        #q_matrix[t_idx+1,:] = q_t + (1/tau)*(1 - (sigma/2) - numpy.exp(q_t)/k)*delta_t + noise_constant*stats.norm.rvs(size=n_reps)
        q_t = q_t_minus_one + (1/tau)*(1 - (sigma/2) - numpy.exp(q_t_minus_one)/k)*delta_t + noise_constant*stats.norm.rvs(size=n_reps)

        # sampling event

        #modulus_t = t_idx % n_grid_points 
        #if t_idx % n_grid_points == 0:
        # supposed to be faster than if statement
        match (t_idx % n_grid_points):
            case 0:
                q_matrix[int(t_idx/n_grid_points),:] = q_t

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




def simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, delta_t = 1, x_0=None):

    # Uses the convolution derived by Dornic et al to generate a trajctory of a migration-birth-drift SDE
    # sampling occurs *daily* like in the human gut timeseries

    # delta_t = time between sampling events (1 day) *NOT* gridpoints
    # we are exactly simulating the FPE, not approximating it via Euler
    

    # expected value of the stationary distribution

    if x_0 == None:
        x_0 = m/r

    # redefine variables for conveinance
    alpha = m
    beta = -1*r
    sigma = numpy.sqrt(2*D)

    x_matrix = numpy.zeros(shape=(n_days+1, n_reps))
    x_matrix[0,:] = x_0

    lambda_ = (2*beta)/((sigma**2) * (numpy.exp(beta*delta_t)-1) )
    mu_ = -1 + (2*alpha)/(sigma**2)
    
    for i in range(n_days):
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

        print(x_matrix[i+1,:10])
        
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
            
            #run_values_j = run_values[run_j_idx]
            run_starts_j = run_starts[run_j_idx]
            run_length_j = run_lengths[run_j_idx]

            # cannot have a sojourn time that is the length of the entire timeseries
            # weird fluctuations here, ignore..
            if (run_length_j >= deviation_x.shape[0]-3):
                continue

            if epsilon != None:
                
                # avoid issue of walk starting at zero
                # inclusive
                start_before = abs(deviation_traj[max([(run_starts_j-1), 0])])
                start_after = abs(deviation_traj[run_starts_j])
                # inclusive
                end_before = abs(deviation_traj[(run_starts_j + run_length_j-1)])
                end_after = abs(deviation_traj[min([(len(deviation_traj)-1), (run_starts_j + run_length_j)])])

                start_before_bool = False
                start_after_bool = False
                end_before_bool = False
                end_after_bool = False
                
                if start_before <= epsilon:
                    start_before_bool = True 

                if start_after <= epsilon:
                    start_after_bool = True 

                if end_before <= epsilon:
                    end_before_bool = True 

                if end_after <= epsilon:
                    end_after_bool = True 

                # continue, not within epsilon at either option
                if ((start_before_bool+start_after_bool) == 0) or ((end_before_bool+end_after_bool) == 0):
                    continue
                    
                # Baldassarri used the first timepoint that reached epsilon 
                # inclusive
                if start_before_bool == True:
                    new_run_start_j = max([(run_starts_j-1), 0])
                else:
                    new_run_start_j = run_starts_j
                
                # first timepoint at end that reached epsilon
                # exclusive
                if end_before == True:
                    new_run_end_j = run_starts_j + run_length_j
                else:
                    new_run_end_j = run_starts_j + run_length_j + 1

                if new_run_start_j == new_run_end_j:
                    continue

                run_deviation_j = numpy.absolute(deviation_traj[new_run_start_j:new_run_end_j])

            else:
                run_deviation_j = numpy.absolute(deviation_traj[run_starts_j:(run_starts_j + run_length_j)])

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

    sigma_range = numpy.logspace(-3, numpy.log10(2), num=10, endpoint=False, base=10)
    k_range = numpy.logspace(2, 6, num=10, endpoint=True, base=10)
    tau_range = numpy.logspace(numpy.log10(0.1), numpy.log10(1000), num=10, endpoint=True, base=10)

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
    m_range = numpy.logspace(numpy.log10(1), numpy.log10(10000), num=10, endpoint=True, base=10)
    r_range = numpy.logspace(numpy.log10(0.0001), numpy.log10(0.01), num=10, endpoint=True, base=10)
    D_range = numpy.logspace(numpy.log10(0.001), numpy.log10(100), num=10, endpoint=True, base=10)

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



def simulate_sojourn_time_vs_cumulative_walk_length(n_days, n_reps, n_slopes=100, min_run_length=10):

    sigma_range = numpy.logspace(-3, numpy.log10(2), num=10, endpoint=False, base=10)
    k_range = numpy.logspace(2, 6, num=10, endpoint=True, base=10)
    tau_range = numpy.logspace(numpy.log10(0.1), numpy.log10(1000), num=10, endpoint=True, base=10)

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

    m_range = numpy.logspace(numpy.log10(1), numpy.log10(10000), num=10, endpoint=True, base=10)
    r_range = numpy.logspace(numpy.log10(0.0001), numpy.log10(0.01), num=10, endpoint=True, base=10)
    D_range = numpy.logspace(numpy.log10(0.001), numpy.log10(100), num=10, endpoint=True, base=10)

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

    m_range = numpy.logspace(numpy.log10(1), numpy.log10(10000), num=10, endpoint=True, base=10)
    r_range = numpy.logspace(numpy.log10(0.0001), numpy.log10(0.01), num=10, endpoint=True, base=10)
    D_range = numpy.logspace(numpy.log10(0.01), numpy.log10(100), num=10, endpoint=True, base=10)

    sigma_range = numpy.logspace(-2, numpy.log10(2), num=10, endpoint=False, base=10)
    k_range = numpy.logspace(2, 6, num=10, endpoint=True, base=10)
    tau_range = numpy.logspace(numpy.log10(0.1), numpy.log10(100), num=10, endpoint=True, base=10)

    #m_range = [m_range[0]]
    #r_range = [r_range[0]]
    #D_range = [D_range[0]]

    #sigma_range = [sigma_range[0]]
    #k_range = [k_range[0]]
    #tau_range = [tau_range[0]]

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
                x_matrix = simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, x_0=mean_gamma)
                x_matrix_sqrt = numpy.sqrt(x_matrix)
                
                # skip if there are any non finite values after square root transform
                if numpy.isfinite(x_matrix_sqrt).all() == False:
                    continue

                run_lengths = calculate_sojourn_time_dist(x_matrix, min_run_length=1)
                run_lengths_sqrt = calculate_sojourn_time_dist(x_matrix_sqrt, min_run_length=1, x_0=mean_square_root_gamma)

                print('Demog', m, r, D)

                dist_dict['demog'][m][r][D] = {}
                dist_dict['demog'][m][r][D]['linear'] = dict(Counter(run_lengths.tolist()))
                dist_dict['demog'][m][r][D]['sqrt'] = dict(Counter(run_lengths_sqrt.tolist()))

                
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
                x_matrix_log = numpy.log(x_matrix)

                if numpy.isfinite(x_matrix_log).all() == False:
                    continue

                run_lengths = calculate_sojourn_time_dist(x_matrix, min_run_length=1)
                run_lengths_log = calculate_sojourn_time_dist(x_matrix_log, min_run_length=1, x_0=mean_log_gamma)

                print('SLM', sigma, k, tau)

                dist_dict['slm'][sigma][k][tau] = {}
                dist_dict['slm'][sigma][k][tau]['linear'] = dict(Counter(run_lengths.tolist()))
                dist_dict['slm'][sigma][k][tau]['log'] = dict(Counter(run_lengths_log.tolist()))


    sys.stderr.write("Saving dictionary...\n")
    with open(sojourn_time_dist_sim_dict_path, 'wb') as outfile:
        pickle.dump(dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")

    # mylist = [key for key, val in mydict.items() for _ in range(val)]
    



if __name__ == "__main__":

    print("Running...")

    #n_days=1000
    #n_reps=100

    simulate_sojourn_time_dist()

  



#x_matrix = simulate_demog_trajectory_dornic(n_days, 10000, 100, 0.1, 1)

#run_lengths_set, rescaled_mean_run_deviation_all = calculate_mean_deviation_pattern_simulation(x_matrix)



#simulate_sojourn_time_vs_cumulative_walk_length(n_days, n_reps, n_slopes=10)


#simulate_cv_sojourn_time(n_days, n_reps)

