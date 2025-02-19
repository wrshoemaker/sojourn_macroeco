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


demog_dict_path = '%sdemog_dict.pickle' % config.data_directory


n_days=1000
n_reps=10000

m_range = numpy.logspace(numpy.log10(1), numpy.log10(10000), num=10, endpoint=True, base=10)
r_range = numpy.logspace(numpy.log10(0.0001), numpy.log10(0.01), num=10, endpoint=True, base=10)
D_range = numpy.logspace(numpy.log10(0.01), numpy.log10(100), num=10, endpoint=True, base=10)



def make_rescaled_demog_dict():

    demog_dict = {}

    for m in m_range:
            
        demog_dict[m] = {}
        
        for r in r_range:

            demog_dict[m][r] = {}

            for D in D_range:

                mean_gamma, cv_gamma = simulation_utils.calculate_mean_and_cv_demog(m, r, D)
                mean_square_root_gamma = stats_utils.expected_value_square_root_gamma(mean_gamma, cv_gamma)

                # unlikely that the square root transfomation will yield useful results
                if numpy.isnan(mean_square_root_gamma) == True:
                    continue

                # initial condition is expected stationary value of square root of gamma rv
                x_matrix = simulation_utils.simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, x_0=mean_gamma)
                #x_matrix_sqrt = simulation_utils.simulate_demog_trajectory_dornic(n_days, n_reps, m, r, D, x_0=mean_gamma)
                # First timepoint has already been transformed....
                #x_matrix_sqrt[1:,:] = numpy.sqrt(x_matrix[1:,:])
                x_matrix_sqrt = numpy.sqrt(x_matrix)
                
                # skip if there are any non finite values after square root transform
                if numpy.isfinite(x_matrix_sqrt).all() == False:
                    continue

                epsilon = 0.1*mean_gamma
                epsilon_sqrt = 0.1*mean_square_root_gamma

                run_length, mean_run_deviation = simulation_utils.calculate_mean_deviation_pattern_simulation(x_matrix, min_run_length=10, min_n_runs=50, epsilon=epsilon)
                # epsilon for square root and use expected value of square root stationary gamma rv as initial condition...
                run_length_sqrt, mean_run_deviation_sqrt = simulation_utils.calculate_mean_deviation_pattern_simulation(x_matrix_sqrt, min_run_length=10, min_n_runs=50, epsilon=epsilon_sqrt, x_0=mean_square_root_gamma)

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




def add_norm_const_to_dict():

    demog_dict = pickle.load(open(demog_dict_path, "rb"))

    for m in demog_dict.keys():

        for r in demog_dict[m].keys():

            for D in demog_dict[m][r].keys():

                for data_type_idx, data_type in enumerate(['linear', 'sqrt']):

                    run_length = numpy.asarray(demog_dict[m][r][D][data_type]['run_length'])

                    s_range = []
                    norm_constant = []

                    for run_length_linear_i_idx, run_length_linear_i in enumerate(run_length):

                        mean_run_deviation_i = numpy.asarray(demog_dict[m][r][D][data_type]['mean_run_deviation'][run_length_linear_i_idx])

                        s_range_i = numpy.linspace(0, 1, num=run_length_linear_i, endpoint=True)
                        norm_constant_i = stats_utils.estimate_normalization_constant(s_range_i, mean_run_deviation_i)

                        s_range.append(s_range_i)
                        norm_constant.append(norm_constant_i)

                    
                    norm_constant = numpy.asarray(norm_constant)
                    slope, intercept = stats_utils.log_log_regression(run_length, norm_constant, min_x=10)

                    demog_dict[m][r][D][data_type]['slope'] = slope
                    demog_dict[m][r][D][data_type]['intercept'] = intercept
                    demog_dict[m][r][D][data_type]['norm_constant'] = norm_constant.tolist()


    sys.stderr.write("Saving dictionary...\n")
    with open(demog_dict_path, 'wb') as outfile:
        pickle.dump(demog_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")

                    



def plot_sojourn_vs_norm_constant(m, r, D):

    demog_dict = pickle.load(open(demog_dict_path, "rb"))

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    for data_type_idx, data_type in enumerate(['linear', 'sqrt']):

        run_length = demog_dict[m][r][D][data_type]['run_length']
        norm_constant = demog_dict[m][r][D][data_type]['norm_constant']
        slope = demog_dict[m][r][D][data_type]['slope']
        intercept = demog_dict[m][r][D][data_type]['intercept']

        run_length = numpy.asarray(run_length)
        norm_constant = numpy.asarray(norm_constant)

        x_log10_range =  numpy.linspace(min(numpy.log10(run_length)) , max(numpy.log10(run_length)) , 10000)
        y_log10_fit_range = (slope*x_log10_range + intercept)

        ax = plt.subplot2grid((1,2), (0,data_type_idx))

        ax.plot(10**x_log10_range, 10**y_log10_fit_range, c='k', lw=2.5, linestyle='-', zorder=2, label="OLS regression slope")
        ax.scatter(run_length, norm_constant, s=10, alpha=0.4, c='b', zorder=1)

        ax.set_xlabel('Sojourn time, ' + r'$T$')
        ax.set_ylabel("Normalization constant, " + r'$N(T)$')

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)

        ax.set_title(data_type, fontsize=12)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssojourn_vs_norm_constant.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_compare_slopes():

    demog_dict = pickle.load(open(demog_dict_path, "rb"))
    demog_dict = demog_dict['demog']

    slopes_linear = []
    slopes_sqrt = []

    cv_x_all = []
    mean_x_all = []
    timescale_all = []
    for m in demog_dict.keys():

        for r in demog_dict[m].keys():

            for D in demog_dict[m][r].keys():
                
                #print(demog_dict[m][r][D].keys())
                slopes_linear.append(demog_dict[m][r][D]['linear']['slope'])
                slopes_sqrt.append(demog_dict[m][r][D]['sqrt']['slope'])

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m, r, D)

                #if demog_dict['demog'][m][r][D]['sqrt']['slope'] < 0.2:
                    #print(round(mean_x, 3), round(cv_x, 3), round(1/r, 3))

                #    print(m, r, D)

                print(m, r, D)

                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)
                timescale_all.append(1/r)


    cv_x_all = numpy.asarray(cv_x_all)
    mean_x_all = numpy.asarray(mean_x_all)
    fractile_cv = sum(cv_x_all>3)/len(cv_x_all)

    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(slopes_linear, slopes_sqrt, s=10, c='k', alpha=0.2)

    min_ = min(slopes_linear+slopes_sqrt)
    max_ = max(slopes_linear+slopes_sqrt)

    ax.plot([min_, max_], [min_, max_], ls=':', lw=2, c='k')

    ax.set_xlabel('Slope, linear', fontsize=12)
    ax.set_ylabel("Slope, square-root", fontsize=12)

    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scompare_slopes_sqrt.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()


    # investigate problematic parameter regimes
    fig = plt.figure(figsize = (12, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    ax_mean = plt.subplot2grid((1,3), (0,0))
    ax_cv = plt.subplot2grid((1,3), (0,1))
    ax_timescale = plt.subplot2grid((1,3), (0,2))

    ax_mean.scatter(mean_x_all, slopes_linear, s=10, c='b', alpha=0.2, label='No transformation')
    ax_mean.scatter(mean_x_all, slopes_sqrt, s=10, c='r', alpha=0.2, label = 'Square-root transformation')
    ax_mean.set_xlabel("Mean", fontsize=12)
    ax_mean.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_mean.set_xscale('log', base=10)
    ax_mean.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')
    ax_mean.legend(loc='upper right')

    ax_cv.scatter(cv_x_all, slopes_linear, s=10, c='b', alpha=0.2)
    ax_cv.scatter(cv_x_all, slopes_sqrt, s=10, c='r', alpha=0.2)
    ax_cv.set_xlabel("CV", fontsize=12)
    ax_cv.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_cv.set_xscale('log', base=10)
    ax_cv.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')


    ax_timescale.scatter(timescale_all, slopes_linear, s=10, c='b', alpha=0.2)
    ax_timescale.scatter(timescale_all, slopes_sqrt, s=10, c='r', alpha=0.2)
    ax_timescale.set_xlabel("Autocorr. timescale (1/birth)", fontsize=12)
    ax_timescale.set_ylabel("Exponent of sojourn time vs. integral", fontsize=12)
    ax_timescale.set_xscale('log', base=10)
    ax_timescale.axhline(y=0.5, lw=2, ls=':', c='k', label='Brownian motion')



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%scv_vs_slope.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_rescaled_deviation(m, r, D):

    demog_dict = pickle.load(open(demog_dict_path, "rb"))

    demog_dict = demog_dict['demog']

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    run_length_all = demog_dict[m][r][D]['linear']['run_length'] + demog_dict[m][r][D]['sqrt']['run_length']

    rgb_blue = cm.Blues(numpy.linspace(0,1,max(run_length_all)))

    for data_type_idx, data_type in enumerate(['linear', 'sqrt']):

        run_length = demog_dict[m][r][D][data_type]['run_length']
        mean_run_deviation = demog_dict[m][r][D][data_type]['mean_run_deviation']

        intercept = demog_dict[m][r][D][data_type]['intercept']
        norm_constant = demog_dict[m][r][D][data_type]['norm_constant']

        ax = plt.subplot2grid((1,2), (0,data_type_idx))

        ax.set_xlim([0,1])

        for run_length_i_idx, run_length_i in enumerate(run_length):

            s_range_i = numpy.linspace(0, 1, num=run_length_i, endpoint=True)
            
            rescaled_mean_run_deviation_i = mean_run_deviation[run_length_i_idx]/(norm_constant[run_length_i_idx]/(10**intercept))
            ax.plot(s_range_i, rescaled_mean_run_deviation_i, alpha=0.4, c=rgb_blue[run_length[run_length_i_idx]-1], lw=1)

            ax.set_xlabel('Rescaled time within sojourn period', fontsize=12)
            ax.set_ylabel("Rescaled mean deviation", fontsize=12)
            ax.set_title(data_type, fontsize=12)

            print(data_type, run_length_i)

        if data_type_idx == 1:

            from matplotlib.colors import Normalize
            cmappable = cm.ScalarMappable(Normalize(0,1), cmap='Blues')

            cbar = plt.colorbar(cmappable, ticks=[0, 1])
            cbar.ax.set_yticklabels([min(run_length_all), max(run_length_all)])  # vertically oriented colorbar
            cbar.set_label('Sojourn time, ' + r'$T$', rotation=270, fontsize=11)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_deviation_demog.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()




def plot_rescaled_deviation_all_params():

    demog_dict = pickle.load(open(demog_dict_path, "rb"))

    demog_dict = demog_dict['demog']
    
    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)

    for data_type_idx, data_type in enumerate(['linear', 'sqrt']):

        ax = plt.subplot2grid((1,2), (0,data_type_idx))
        ax.set_xlabel('Rescaled time within sojourn period', fontsize=12)
        ax.set_ylabel("Rescaled mean deviation", fontsize=12)
        ax.set_title(data_type, fontsize=12)

        for m in demog_dict.keys():

            for r in demog_dict[m].keys():

                for D in demog_dict[m][r].keys():

                    run_length = demog_dict[m][r][D][data_type]['run_length']
                    mean_run_deviation = demog_dict[m][r][D][data_type]['mean_run_deviation']

                    intercept = demog_dict[m][r][D][data_type]['intercept']
                    norm_constant = demog_dict[m][r][D][data_type]['norm_constant']

                    s_range_all = []
                    rescaled_mean_run_deviation_all = []

                    for run_length_i_idx, run_length_i in enumerate(run_length):

                        s_range_i = numpy.linspace(0, 1, num=run_length_i, endpoint=True)
                        rescaled_mean_run_deviation_i = mean_run_deviation[run_length_i_idx]/(norm_constant[run_length_i_idx]/(10**intercept))

                        s_range_all.extend(s_range_i.tolist())
                        rescaled_mean_run_deviation_all.extend(rescaled_mean_run_deviation_i.tolist())

                    
                    s_range_all = numpy.asarray(s_range_all)
                    rescaled_mean_run_deviation_all = numpy.asarray(rescaled_mean_run_deviation_all)

                    hist_x_all, bin_edges_x_all = numpy.histogram(s_range_all, density=True, bins=50)
                    #bins_x_all = [0.5 * (bin_edges_x_all[i] + bin_edges_x_all[i+1]) for i in range(0, len(bin_edges_x_all)-1 )]
                    bins_x_all_to_keep = []
                    bins_y = []
                    for i in range(0, len(bin_edges_x_all)-1 ):
                        y_i = rescaled_mean_run_deviation_all[(s_range_all>=bin_edges_x_all[i]) & (s_range_all<bin_edges_x_all[i+1])]
                        bins_x_all_to_keep.append(bin_edges_x_all[i])
                        bins_y.append(numpy.mean(y_i))


                    bins_x_all_to_keep = numpy.asarray(bins_x_all_to_keep)
                    bins_y = numpy.asarray(bins_y)

                    ax.plot(bins_x_all_to_keep, bins_y, lw=0.5, ls='-', c='k', alpha=0.3)

                    #if (data_type == 'sqrt') and (max(bins_y) > 8.5):

                    #    print(m,r,D)

                    if (data_type == 'linear') and (max(bins_y) > 40000):

                        print(m,r,D)


    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_deviation_all_params_demog.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()



def plot_rescaled_same_sojourn_time(sojourn_time=10, delta_t=1):

    demog_dict = pickle.load(open(demog_dict_path, "rb"))
    demog_dict = demog_dict['demog']

    s_range = numpy.linspace(0, 1, num=sojourn_time, endpoint=True)

    mean_x_all = []
    cv_x_all = []
    timescale_x_all = []
    D_all = []
    for m in demog_dict.keys():

        for r in demog_dict[m].keys():

            for D in demog_dict[m][r].keys():

                if (sojourn_time not in (demog_dict[m][r][D]['linear']['run_length'])) or (sojourn_time not in (demog_dict[m][r][D]['sqrt']['run_length'])):
                        continue

                mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)
                mean_x_all.append(mean_x)
                cv_x_all.append(cv_x)
                timescale_x_all.append(1/r)
                D_all.append(D)

    #rgb_blue = cm.Blues(numpy.logspace(0, 1, max(run_length_all), base=10))

    fig = plt.figure(figsize = (8, 4)) #
    fig.subplots_adjust(bottom= 0.1,  wspace=0.15)
    fig.suptitle('Sojourn time ' + r'$T=$' + str(sojourn_time), fontsize=12)

    D_all = numpy.asarray(D_all)
    noise_constant = numpy.sqrt(2*D_all*delta_t)
    cmap = cm.ScalarMappable(norm = colors.LogNorm(min(noise_constant), max(noise_constant)), cmap = plt.get_cmap('Blues'))
    #cmap = cm.ScalarMappable(norm = colors.Normalize(min(cv_x_all), max(cv_x_all)), cmap = plt.get_cmap('Blues'))

    data_type_title_dict = {'linear':'No data transformation', 'sqrt':'Square-root transformation'}

    for data_type_idx, data_type in enumerate(['linear', 'sqrt']):

        ax = plt.subplot2grid((1,2), (0,data_type_idx))
        ax.set_xlabel('Rescaled time within sojourn period, ' + r'$t$', fontsize=12)
        ax.set_ylabel("Rescaled mean deviation, " + r'$\left < x(t) - x(0) \right >_{T}$', fontsize=12)
        ax.set_title(data_type_title_dict[data_type], fontsize=12)
        #ax.set_yscale('log', base=10)

        for m in demog_dict.keys():

            for r in demog_dict[m].keys():

                for D in demog_dict[m][r].keys():

                    if (sojourn_time not in (demog_dict[m][r][D]['linear']['run_length'])) or (sojourn_time not in (demog_dict[m][r][D]['sqrt']['run_length'])):
                        continue

                    run_length = demog_dict[m][r][D][data_type]['run_length']
                    mean_run_deviation = demog_dict[m][r][D][data_type]['mean_run_deviation']

                    intercept = demog_dict[m][r][D][data_type]['intercept']
                    norm_constant = demog_dict[m][r][D][data_type]['norm_constant']

                    run_length = numpy.asarray(run_length)
                    sojourn_time_idx = numpy.where(run_length == sojourn_time)[0][0]

                    mean_run_deviation_target = numpy.asarray(mean_run_deviation[sojourn_time_idx])

                    mean_x, cv_x = simulation_utils.calculate_mean_and_cv_demog(m,r,D)
                    #mean_run_deviation_target_rescaled_by_mean = mean_run_deviation_target/mean_x

                    ax.plot(s_range, mean_run_deviation_target, lw=0.5, ls='-', c=cmap.to_rgba(numpy.sqrt(2*D*delta_t)), alpha=0.3)

                    #if (max(mean_run_deviation_target) > 20) and (data_type == 'sqrt'):
                    #    print(m, r, D)


                    #for run_length_i_idx, run_length_i in enumerate(run_length):

                    
                    #rescaled_mean_run_deviation_i = mean_run_deviation[run_length_i_idx]/(norm_constant[run_length_i_idx]/(10**intercept))

                    #s_range_all.extend(s_range_i.tolist())
                    #rescaled_mean_run_deviation_all.extend(rescaled_mean_run_deviation_i.tolist())

        #if data_type == 'linear':
        ax.set_yscale('log', base=10)



    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%srescaled_same_sojourn_time.png" % (config.analysis_directory)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()





if __name__ == "__main__":

    print("Running...")

    #plot_rescaled_same_sojourn_time(sojourn_time=25)

    #make_rescaled_demog_dict()

    # figs to make

    #add_norm_const_to_dict()
    plot_compare_slopes()
    #plot_sojourn_vs_norm_constant(2.7825594022071245, 0.003593813663804626, 100.0)

    # cool sqrt.
    #plot_rescaled_deviation(7.742636826811269, 0.002154434690031882, 100.0)

    # weird linear
    #plot_rescaled_deviation(10000.0, 0.0001, 100.0)

    #plot_rescaled_deviation_all_params()

