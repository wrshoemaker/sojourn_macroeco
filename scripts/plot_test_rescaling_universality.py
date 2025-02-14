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

#import functools
#import operator

import data_utils
import plot_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import simulation_utils


n_days=1000
#n_reps=10000
n_reps=1000000


D = 1
delta_t=1
#x_matrix = simulation_utils.simulate_demog_trajectory_dornic(n_days, n_reps, 100, 0.1, 1)
#x_matrix = simulation_utils.simulate_brownian_trajectory(n_days, n_reps, D, 0, delta_t=delta_t)

# has units L, 10% of per-timestep deviation
epsilon = 0.1*numpy.sqrt(D*delta_t)


#run_length_all_final, rescaled_mean_run_deviation_all = simulation_utils.calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=30, epsilon=epsilon)

#numpy.save('%srun_length_all_final' % config.data_directory, run_length_all_final)
#numpy.save('%srescaled_mean_run_deviation_all' % config.data_directory, rescaled_mean_run_deviation_all)


run_length_all_final = numpy.load('%srun_length_all_final.npy' % config.data_directory, allow_pickle=True)
rescaled_mean_run_deviation_all = numpy.load('%srescaled_mean_run_deviation_all.npy' % config.data_directory, allow_pickle=True)


# check scaling
norm_factor_all = []

for rescaled_mean_run_deviation_i_idx, rescaled_mean_run_deviation_i in enumerate(rescaled_mean_run_deviation_all):
    
    s_range = numpy.linspace(0, 1, num=len(rescaled_mean_run_deviation_i), endpoint=True)
    #norm_factor = integrate.simps(rescaled_mean_run_deviation_i, s_range)
    norm_factor = integrate.simps(rescaled_mean_run_deviation_i, s_range)
    norm_factor_all.append(norm_factor)

norm_factor_all = numpy.asarray(norm_factor_all)



run_length_min = 10

run_length_all_final_filter = run_length_all_final[run_length_all_final >= run_length_min]
norm_factor_all_filter = norm_factor_all[run_length_all_final >= run_length_min]

slope, intercept, r_valuer_value, p_value, std_err = stats.linregress(numpy.log10(run_length_all_final_filter), numpy.log10(norm_factor_all_filter))

fig, ax = plt.subplots(figsize=(6,4))

x_log10_range =  numpy.linspace(min(numpy.log10(run_length_all_final)) , max(numpy.log10(run_length_all_final)) , 10000)
y_log10_fit_range = (slope*x_log10_range + intercept)

ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
ax.scatter(run_length_all_final, norm_factor_all, s=10, alpha=0.4, c='b', zorder=1)

print(slope, std_err)

ax.set_xlabel('Sojourn time, ' + r'$T$')
ax.set_ylabel("Normalization constant, " + r'$N(T)$')



ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)

fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stest_factor.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()



rgb_blue = cm.Blues(numpy.linspace(0,1,max(run_length_all_final)))

fig, ax = plt.subplots(figsize=(6,4))

for rescaled_mean_run_deviation_i_idx, rescaled_mean_run_deviation_i in enumerate(rescaled_mean_run_deviation_all):

    s_range = numpy.linspace(0, 1, num=len(rescaled_mean_run_deviation_i), endpoint=True)

    
    #print(normalization_factor, sum(rescaled_mean_run_deviation_i))

    #rescaled_mean_run_deviation_i_ = rescaled_mean_run_deviation_i/(len(rescaled_mean_run_deviation_i)**(0.5))
    #rescaled_mean_run_deviation_i_ = rescaled_mean_run_deviation_i_*0.75

    rescaled_mean_run_deviation_i_ = rescaled_mean_run_deviation_i/(norm_factor_all[rescaled_mean_run_deviation_i_idx]/(10**intercept))

    #print(len(rescaled_mean_run_deviation_i), (len(rescaled_mean_run_deviation_i)**(0.5))/0.75, sum(rescaled_mean_run_deviation_i))

    ax.plot(s_range, rescaled_mean_run_deviation_i_, alpha=0.4, c=rgb_blue[run_length_all_final[rescaled_mean_run_deviation_i_idx]-1], lw=1)



s_theory_range = numpy.linspace(0, 1, num=1000, endpoint=True)
x_theory = numpy.sqrt(s_theory_range*(1-s_theory_range))*numpy.sqrt(8/numpy.pi)


ax.plot(s_theory_range, x_theory, alpha=1, c='k', lw=2, label='Theory')
ax.set_xlabel('Rescaled time within sojourn period, ' + r'$\frac{t}{T}$')
ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T} \cdot T^{-\frac{1}{2}} $')

#sm = plt.cm.ScalarMappable(cmap=rgb_blue)
#from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ax.legend(loc='upper left', fontsize=10)

cmappable = cm.ScalarMappable(Normalize(0,1), cmap='Blues')

#cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
#cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

cbar = plt.colorbar(cmappable, ticks=[0, 1])
cbar.ax.set_yticklabels([min(run_length_all_final), max(run_length_all_final)])  # vertically oriented colorbar
cbar.set_label('Sojourn time, ' + r'$T$', rotation=270, fontsize=11)


#ax.plot()

fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%stest_rescaling_universality.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()