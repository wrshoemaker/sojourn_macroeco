import pickle
import sys
import numpy
import data_utils
import plot_utils
import config

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import loggamma, mode, linregress
from statsmodels.stats.multitest import fdrcorrection



n_rows = len(data_utils.dataset_all)
n_cols = 4


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))


fig = plt.figure(figsize = (16, 12)) #
fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
fig.suptitle("Mean relative abundance vs. CV of relative abundance", fontsize=24,  fontweight='bold', y=0.95)  # adjust y to move title up/down


ax_all = []
r_value_all = []
slope_all = []
intercept_all = []
pvalue_all = []
x_range_all = []

for dataset_idx, dataset in enumerate(data_utils.dataset_all):

    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
    host_all = list(mle_dict[dataset].keys())
    host_all.sort()

    for host_idx, host in enumerate(host_all):

        ax = plt.subplot2grid((n_rows, n_cols), (dataset_idx, host_idx))

        x_mean_all = []
        x_var_all = []
        for key, value in mle_dict[dataset][host].items():

            rel_abundance = numpy.asarray(value['rel_abundance'])

            x_mean = value['x_mean']
            #x_var = (value['x_std'])**2
            x_cv = value['x_std']/x_mean

            ax.scatter(x_mean, x_cv, color=plot_utils.host_color_dict[dataset][host], alpha=0.8, s=10)

            x_mean_all.append(x_mean)
            x_var_all.append(x_cv) 
        

        if host_idx == 0:
            ax.set_ylabel('CV rel. abundance, ' + r'$\mathrm{CV}_{x_{i}}$', fontsize=12)
        
        # x-label
        if (dataset_idx == len(data_utils.dataset_all)-1) or ((host_idx >= 2) and (dataset_idx==1)):
            ax.set_xlabel('Mean relative abundance, ' + r'$\bar{x}_{i}$', fontsize=12)


        #ax.set_title(plot_utils.host_name_dict[dataset][host], fontsize=12)
        ax.set_title(plot_utils.label_dataset_host(dataset, host), fontsize=12)
        log10_x_mean_all = numpy.log10(x_mean_all)
        #log10_x_var_all = numpy.log10(x_var_all)

        slope, intercept, r_value, pvalue, std_err = linregress(log10_x_mean_all, x_var_all)
        #intercept_exponent_2 = numpy.mean(log10_x_var_all) - 2*numpy.mean(log10_x_mean_all)

        slope_all.append(slope)
        r_value_all.append(r_value)
        intercept_all.append(intercept)
        pvalue_all.append(pvalue)
        x_range_all.append(numpy.linspace(min(log10_x_mean_all), max(log10_x_mean_all), 10000))
        ax_all.append(ax)


        #if p_value < 0.05:

        #    x_range =  numpy.linspace(min(log10_x_mean_all), max(log10_x_mean_all), 10000)
        #    y_fit_range = slope*x_range_ + intercept
        #    #y_fit_range_exponent_2 = 2*x_range_ + intercept_exponent_2

        #    ax.plot(10**x_range_, y_fit_range, ls='-', lw=2.5, c='k', label='Fitted exponent = %0.4f' % slope)
        #    #ax.plot(10**x_range_, 10**y_fit_range_exponent_2, ls=':', lw=2.5, c='k', label='Exponent = 2')
        #    #ax.set_xlim([min(x_mean_all), max(x_mean_all)])
        #    ax.legend(loc='upper left', fontsize=10)

        

        #if (host_idx == 0) and (dataset_idx==0):

pvalue_all = numpy.asarray(pvalue_all)
reject_all, pvalue_corrected_all = fdrcorrection(pvalue_all, alpha=0.05)

#x_text_all = []
y_text_all = [0.8, 0.7, 0.8, 0.7, 0.8, 0.85, 0.33, 0.8]
x_text_delta_all = [0, 0, 0, 0.47, 0, 0, 0, 0.47]

for ax_idx, ax in enumerate(ax_all):

    pvalue_corrected = pvalue_corrected_all[ax_idx]

    

    ax.text(0.05 + x_text_delta_all[ax_idx], y_text_all[ax_idx], r'$\rho_{\mathrm{Pearson}}^{2} = $' + str(plot_utils.round_sig(r_value_all[ax_idx]**2)), fontsize=11, transform=ax.transAxes)
    ax.text(0.22 + x_text_delta_all[ax_idx], y_text_all[ax_idx] - 0.07, r'$P  = $' + str(plot_utils.round_sig(pvalue_corrected)), fontsize=11, transform=ax.transAxes)

    if pvalue_corrected < 0.05:

        x_range = x_range_all[ax_idx]

        y_fit_range = slope_all[ax_idx]*x_range + intercept_all[ax_idx]
        ax.plot(10**x_range, y_fit_range, ls='-', lw=2.5, c='k')#, label='Fitted exponent = %0.4f' % slope)

        

    ax.set_xscale('log', base=10)

       



fig.subplots_adjust(hspace=0.25, wspace=0.25)
fig_name = "%smean_vs_cv.png" % (config.analysis_directory)
fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
plt.close()