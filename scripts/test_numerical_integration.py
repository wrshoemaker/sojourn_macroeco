


import numpy
from scipy.integrate import quad, quadrature, romberg



def integrand(x, x_0, D, t):
    return (numpy.sqrt(numpy.pi*D*t*4)**-1) * (numpy.exp(-1*((x-x_0)**2) / (4*D*t)) + numpy.exp(-1* ((x+x_0)**2) / (4*D*t)) )


I = quad(integrand, 0, numpy.inf, args=(50, 1, 5))

print(I)