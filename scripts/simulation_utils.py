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
from scipy.special import digamma, gamma, erf
from scipy import interpolate

import data_utils


numpy.random.seed(123456789)

slope_dict_path = '%sfluctuation_slope_dict.pickle' %config.data_directory
sojourn_moment_dict_path = '%ssojourn_moments_dict.pickle' %config.data_directory


def calculate_mean_and_cv_slm(k, sigma):

    mean_ = k*(1-(sigma/2))
    cv_ = numpy.sqrt(sigma/(2-sigma))

    return mean_, cv_


def calculate_mean_and_cv_demog(m, r, D):

    mean_ = m/r
    cv_ = numpy.sqrt(D/m)

    return mean_, cv_




def simulate_slm_trajectory(n_days=1000, n_reps=100, k=10000, sigma=1, tau=7, n_grid_points=1000, init_log=False, return_log=False):

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
    
    # +1 for initial condition
    q_matrix = numpy.zeros(shape=(n_iter+1, n_reps))
    q_matrix[0,:] = q_0

    z = numpy.random.randn(n_iter, n_reps)

    for t_idx in range(n_iter):

        q_t = q_matrix[t_idx,:]
        # we can use t_idx for z because it has one less column than q_matrix
        q_matrix[t_idx+1,:] = q_t + (1/tau)*(1 - (sigma/2) - numpy.exp(q_t)/k)*delta_t + numpy.sqrt(sigma*delta_t/tau)*z[t_idx,:]


    # select samples you want to keep
    sample_idx = numpy.arange(0, n_iter, n_grid_points)
    sample_idx = numpy.append(sample_idx, n_iter)
    q_matrix_sample = q_matrix[sample_idx,:]

    if return_log == True:
        matrix_sample_final = q_matrix_sample

    else:
        matrix_sample_final = numpy.exp(q_matrix_sample)


    return matrix_sample_final




def simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, delta_t = 1):

    # Uses the convolution derived by Dornic et al to generate a trajctory of a migration-birth-drift SDE
    # sampling occurs *daily* like in the human gut timeseries

    # delta_t = time between sampling events (1 day) *NOT* gridpoints
    # we are exactly simulating the FPE, not approximating it via Euler
    

    # expected value of the stationary distribution
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



def calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=10, epsilon=None):

    # min_n_runs = minimum number of replicate trajectories belonging to a given sojourn time to be included in the analysis

    deviation_x = x_matrix[1:,:] - x_matrix[0,:]

    # relative deviation from origin: epsilon = (x(t) - x(0))/x(0)
    # Absolute value of the relative deviation must be within +/- epsilon at start and end of sojourn period
    #epsilon = 0.1
    #x_epsilon = x_matrix[0,0]*epsilon

    run_lengths_all = []
    run_sum_all = []
    run_deviation_all = []

    for deviation_traj in deviation_x.T:

        run_values, run_starts, run_lengths = data_utils.find_runs(deviation_traj>0, min_run_length=min_run_length)

        #run_lengths_all.append(run_lengths)

        for run_j_idx in range(len(run_values)):
            
            #run_values_j = run_values[run_j_idx]
            run_starts_j = run_starts[run_j_idx]
            run_length_j = run_lengths[run_j_idx]

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


            run_lengths_all.append(len(run_deviation_j))
            run_sum_all.append(sum(run_deviation_j))
            run_deviation_all.append(run_deviation_j)


    #run_lengths_all = numpy.concatenate(run_lengths_all).ravel()
    run_lengths_all = numpy.asarray(run_lengths_all)
    run_sum_all = numpy.asarray(run_sum_all, dtype=object)
    run_deviation_all = numpy.asarray(run_deviation_all, dtype=object)

    # sorts the set
    #run_lengths_set = numpy.asarray(list(set(run_lengths_all.tolist())))
    run_lengths_set = numpy.unique(run_lengths_all)
    rescaled_mean_run_deviation_all = []
    run_length_all_final = []
    for run_length_i in run_lengths_set:

        idx_i = (run_lengths_all==run_length_i)

        # Insufficient number of replicates
        if sum(idx_i) < min_n_runs:
            continue

        #rescaling_constant_i = numpy.mean(run_sum_all[idx_i])
        
        # slice and form into matrix.
        run_deviation_matrix_i = numpy.vstack(run_deviation_all[idx_i])
        # try rescaling by sum within each replicate
        #print(run_length_i, len(numpy.sum(run_deviation_matrix_i, axis=1)))
        #run_deviation_matrix_i = run_deviation_matrix_i/numpy.sum(run_deviation_matrix_i, axis=1)
        #run_deviation_matrix_i = ((run_deviation_matrix_i.T)/numpy.sum(run_deviation_matrix_i, axis=1)).T

        mean_run_deviation_i = numpy.mean(run_deviation_matrix_i, axis=0)
        #rescaled_mean_run_deviation_i = mean_run_deviation_i#/sum(mean_run_deviation_i)
        run_length_all_final.append(run_length_i)
        rescaled_mean_run_deviation_all.append(mean_run_deviation_i)

    # should be sorted
    run_length_all_final = numpy.asarray(run_length_all_final)
    rescaled_mean_run_deviation_all = numpy.asarray(rescaled_mean_run_deviation_all, dtype=object)

    return run_length_all_final, rescaled_mean_run_deviation_all
    




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






if __name__ == "__main__":

    print("Running...")

    n_days=1000
    n_reps=100

    test_x = simulate_brownian_trajectory_semi_infinite(n_days, n_reps, D=1, x_0=1000, delta_t=1)

    print(test_x)




#x_matrix = simulate_demog_trajectory_dornic(n_days, 10000, 100, 0.1, 1)

#run_lengths_set, rescaled_mean_run_deviation_all = calculate_mean_deviation_pattern_simulation(x_matrix)



#simulate_sojourn_time_vs_cumulative_walk_length(n_days, n_reps, n_slopes=10)


#simulate_cv_sojourn_time(n_days, n_reps)

