

import numpy
from scipy.special import gamma




def predict_sojourn_dist_slm(t_max, k, sigma, tau, x_0):

    # Eq. 8 in  Kearney and Martin 2021 J. Phys. A: Math. Theor. 54 055002

    y_0 = numpy.log(x_0) - numpy.log(k) - numpy.log(1- (sigma/2))
    gamma_ = (1/tau)*(1 - (sigma/2))
    alpha = y_0*numpy.sqrt(2*gamma_ / (sigma/tau) )

    t_range = numpy.arange(1, t_max+1)

    #t_tilde = gamma*t_range
    
    # ignore exponential term as we are assuming alpha -> 0
    prob_t = numpy.sqrt(2/numpy.pi) * alpha * numpy.exp(-gamma_*t_range) * ((1 - numpy.exp(-2*gamma_*t_range))**(-3/2)) * numpy.exp( -1 * (alpha**2) * numpy.exp(-2*gamma_*t_range) / (2*(1 - numpy.exp(-2*gamma_*t_range))) )
    
    return t_range, prob_t



def predict_sojourn_dist_bdm(t_max, m, psi, tau, x_0):

    theta = 2*m/(psi**2)
    y_0 = (2*x_0)/(psi**2)

    #print(y_0**(1-theta))
    print(theta, gamma(1-theta))

    t_range = numpy.arange(1, t_max+1)
    print(((y_0**(1-theta))/(gamma(1-theta))))
    prob_t = ((y_0**(1-theta))/(gamma(1-theta))) * numpy.exp(-1*y_0 * numpy.exp(-t_range)  / (1 - numpy.exp(-t_range)) ) * ((numpy.exp(-t_range)/ (1 - numpy.exp(-t_range)) ) ** (-1*theta)) (numpy.exp(-1*t_range) / ((1 - numpy.exp(-1*t_range))**2) )

    return t_range, prob_t

