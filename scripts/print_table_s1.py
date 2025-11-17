

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


mle_dict_path_old = '%smle_dict_old.pickle' %config.data_directory
mle_dict = pickle.load(open(mle_dict_path_old, "rb"))



max_delta_t = 20
min_n_sampes = 60



import data_utils


#read_counts_all_sort_filter, host_all_sort_filter, days_all_sort_filter, asv_all_sort = data_utils.get_dada2_data('poyet_et_al', 'gut')

#print(host_all_sort_filter)

#print(days_all_sort_filter[host_all_sort_filter=='am'])

#print(host_all_sort_filter)

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
#print(days)

delta_t = days[1:] - days[:-1]

mean_delta_t = numpy.mean(delta_t)

std_delta_t = numpy.std(delta_t)


#print(delta_t)

#print(days)
print(delta_t)


start_day_idx = 26
start_day = days[start_day_idx]
end_day_idx = 137
end_day = days[end_day_idx]

#print(start_day)

print(end_day)
#print(delta_t[start_day_idx-1])

#print(delta_t[end_day_idx-1])
#print(days[end_day_idx])


#print(delta_t)
delta_t_split, cut_points = split_on_threshold(delta_t, max_delta_t)

#print(delta_t_split)


#print(delta_t[cut_points[0]+1: cut_points[1]])

days_to_keep = days[cut_points[0]+1: cut_points[1]+1]

#print(days_to_keep)
#print(days_to_keep[1:] - days_to_keep[:-1])

# first interval (days)
# [150, 455]

# final interval (days)
# this is because # days is conditioned on a reference start day
# that start day is determined by the set of hosts for a given dataset
# [104, 406]


#print(delta_t_split)
# print(days_to_keep)
    

