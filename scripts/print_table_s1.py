

import pickle
import sys
import numpy
import data_utils
import stats_utils
import plot_utils
import config

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy import stats


mle_dict = pickle.load(open(data_utils.mle_dict_path, "rb"))

max_delta_t = 20
min_n_sampes = 60



import data_utils


read_counts_all_sort_filter, host_all_sort_filter, days_all_sort_filter, asv_all_sort = data_utils.get_dada2_data('poyet_et_al', 'gut')


print(host_all_sort_filter)


def split_on_threshold(arr, thresh):

    cut_points = numpy.where(arr > thresh)[0]

    if len(cut_points) == 0:
        return [arr]   # nothing to split

    subarrays = []
    start = 0

    for idx in cut_points:
        if idx > start:
            subarrays.append(arr[start:idx])
        start = idx + 1

    if start < len(arr):
        subarrays.append(arr[start:])

    return subarrays, cut_points


#for dataset_idx, dataset in enumerate(data_utils.dataset_all):

#    sys.stderr.write("Analyzing dataset %s.....\n" % dataset)
#    host_all = list(mle_dict[dataset].keys())
#    host_all.sort()

    #for host in host_all:

    #    print(host)

asv_all = list(mle_dict['poyet_et_al']['am'].keys())

days = numpy.asarray(mle_dict['poyet_et_al']['am'][asv_all[0]]['days'])

delta_t = days[1:] - days[:-1]

mean_delta_t = numpy.mean(delta_t)

std_delta_t = numpy.std(delta_t)


delta_t_split, cut_points = split_on_threshold(delta_t, max_delta_t)


#print(delta_t[cut_points[0]+1: cut_points[1]])

days_to_keep = days[cut_points[0]+1: cut_points[1]+1]

# [150, 455]

#print(delta_t_split)
# print(days_to_keep)
    

