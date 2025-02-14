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

param_dict = pickle.load(open(simulation_utils.slope_dict_path, "rb"))

fig = plt.figure(figsize = (8.5, 4)) #
fig.subplots_adjust(bottom= 0.15)
gs = gridspec.GridSpec(nrows=1, ncols=2)

ax_slm = fig.add_subplot(gs[0, 0])
ax_demog = fig.add_subplot(gs[0, 1])


mean_slm = []
diff_slm = []
alpha_slm = []

mean_demog = []
diff_demog = []
alpha_demog = []

for sigma in param_dict['slm'].keys():
            
    for k in param_dict['slm'][sigma].keys():

        for tau in param_dict['slm'][sigma][k].keys():

            #if numpy.mean(param_dict['slm'][sigma][k][tau]['std_log_sojourn_time']) < 0.5:
            #    continue

            mean_slm.append( k *(1 - sigma/2))
            diff_slm.append(numpy.sqrt(sigma/tau))
            alpha_slm.append(numpy.mean(param_dict['slm'][sigma][k][tau]['slope']))
            

for m in param_dict['demog'].keys():
            
    for r in param_dict['demog'][m].keys():

        for D in param_dict['demog'][m][r].keys():

            #))

            #if numpy.mean(param_dict['demog'][m][r][D]['std_log_sojourn_time']) < 0.5:
            #    continue

            mean_demog.append(m/r)
            diff_demog.append(2*D)
            alpha_demog.append(numpy.mean(param_dict['demog'][m][r][D]['slope']))



mean_slm = numpy.asarray(mean_slm)
diff_slm = numpy.asarray(diff_slm)
alpha_slm = numpy.asarray(alpha_slm)

mean_demog = numpy.asarray(mean_demog)
diff_demog = numpy.asarray(diff_demog)
alpha_demog = numpy.asarray(alpha_demog)

to_keep_idx =  (alpha_slm>0) & (alpha_demog>0)# & (alpha_slm<0) & (alpha_demog>0)
to_keep_slm_idx =  (alpha_slm>0)
to_keep_demog_idx =  (alpha_demog>0)
mean_slm = mean_slm[to_keep_slm_idx]
diff_slm = diff_slm[to_keep_slm_idx]
alpha_slm = alpha_slm[to_keep_slm_idx]

mean_demog = mean_demog[to_keep_demog_idx]
diff_demog = diff_demog[to_keep_demog_idx]
alpha_demog = alpha_demog[to_keep_demog_idx]

ax_slm.scatter(diff_slm, alpha_slm, alpha=0.7, s=40, edgecolors='k', c=mean_slm, cmap='Blues', norm=colors.LogNorm(vmin=min(mean_slm + mean_demog), vmax=max(mean_slm + mean_demog)), zorder=2)
ax_demog.scatter(diff_demog, alpha_demog, alpha=0.7, s=40, edgecolors='k', c=mean_demog, cmap='Blues', norm=colors.LogNorm(vmin=min(mean_slm + mean_demog), vmax=max(mean_slm + mean_demog)), zorder=2)

#ax_slm.scatter(mean_slm, diff_slm, alpha=1, s=40, edgecolors='k', c=alpha_slm, cmap='Reds', norm=colors.Normalize(vmin=min(alpha_slm + alpha_demog), vmax=max(alpha_slm + alpha_demog)), zorder=2)
#ax_demog.scatter(mean_demog, diff_demog, alpha=1, s=40, edgecolors='k', c=alpha_demog, cmap='Reds', norm=colors.Normalize(vmin=min(alpha_slm + alpha_demog), vmax=max(alpha_slm + alpha_demog)), zorder=2)

ax_slm.set_xscale('log', base=10)
#ax_slm.set_yscale('log', base=10)

ax_demog.set_xscale('log', base=10)
#ax_demog.set_yscale('log', base=10)


ax_slm.set_xlabel('Noise constant, ' + r'$\sqrt{\frac{\sigma}{\tau}}$', fontsize=10)
ax_demog.set_xlabel('Noise constant, ' + r'$2 \cdot D$', fontsize=10)

ax_slm.set_ylabel("Slope b/w sojourn time and cumul. walk length, " + r'$\alpha$', fontsize=10)
ax_demog.set_ylabel("Slope b/w sojourn time and cumul. walk length, " + r'$\alpha$', fontsize=10)

ax_slm.set_title("SLM", fontsize=11)
ax_demog.set_title("Demographic noise", fontsize=11)

#ax_slm.set_xlim([0.0006, 0.5])
#ax_demog.set_xlim([0.0006, 0.5])

ax_slm.set_ylim([0.7, 1.6])
ax_demog.set_ylim([0.7, 1.6])

ax_slm.axhline(y=0.74, ls='--', lw=2, c='k')
ax_slm.axhline(y=1.5, ls='--', lw=2, c='k', label = 'Lowest and highest empirical ' + r'$\alpha$')

ax_demog.axhline(y=0.74, ls='--', lw=2, c='k')
ax_demog.axhline(y=1.5, ls='--', lw=2, c='k', label = 'Lowest and highest empirical ' + r'$\alpha$')

ax_slm.legend(loc='lower left', fontsize=9)


fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%sslope.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()
