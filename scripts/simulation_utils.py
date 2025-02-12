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


numpy.random.seed(123456789)


def simulate_slm_trajectory(n_days=1000, n_reps=100, K = 10000, sigma = 1, tau = 7, n_grid_points=1000, init_log=False, return_log=False):

    # n_days = length of trajectory
    # n_reps = number of replicate trajectories
    # tau = generation timescale of SLM
    # Two timescales: 1) simulation timescale, 2) observation timescale
    # Dimulation timescale = \Delta t = \tau/n_grid_points
    # Observation timesca = daily sampling (set by n_days)
    # init_log = Whether you are want the initial condition to the expected value of the *log* of x. 
    # Useful if you are analyzing log transformed empirical data and need simulations

    # SLM
    x_bar_slm = K*(1- (sigma/2))
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
        q_matrix[t_idx+1,:] = q_t + (1/tau)*(1 - (sigma/2) - numpy.exp(q_t)/K)*delta_t + numpy.sqrt(sigma*delta_t/tau)*z[t_idx,:]


    # select samples you want to keep
    sample_idx = numpy.arange(0, n_iter, n_grid_points)
    sample_idx = numpy.append(sample_idx, n_iter)
    q_matrix_sample = q_matrix[sample_idx,:]

    if return_log == True:
        matrix_sample_final = q_matrix_sample

    else:
        matrix_sample_final = numpy.exp(q_matrix_sample)


    return matrix_sample_final



x_trajectory_slm = simulate_slm_trajectory()

