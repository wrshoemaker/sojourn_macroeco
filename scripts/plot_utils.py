import numpy
from matplotlib import cm
import matplotlib as mpl
from matplotlib import colors



# taxonomic hierarchy colors
cmap_offset = int(0.2*16)
# +cmap_offset

#FF6347
#87CEEB

rgb_blue_phylo = cm.Blues(numpy.linspace(0,1,100+3))
rgb_blue_phylo = mpl.colors.ListedColormap(rgb_blue_phylo[cmap_offset:,:-1])

def build_rgb_blue_phylo(n):

    rgb_blue_phylo = cm.Blues(numpy.linspace(0,1,n+3))
    rgb_blue_phylo = mpl.colors.ListedColormap(rgb_blue_phylo[cmap_offset:,:-1])

    return rgb_blue_phylo


color_radius=2



host_color_dict = {'david_et_al': {'DonorA_post_travel': 'dodgerblue', 'DonorA_pre_travel': 'royalblue', 'DonorB_post_travel': 'cornflowerblue', 'DonorB_pre_travel': 'steelblue'}, 'poyet_et_al': {'ae': 'orangered', 'am': 'darkred', 'an': 'maroon', 'ao': 'firebrick'}, 'caporaso_et_al': {'F4': 'seagreen', 'M3': 'darkgreen'}}

dataset_color_dict = {'david_et_al': 'royalblue', 'poyet_et_al': 'firebrick', 'caporaso_et_al': 'darkgreen'}


data_type_title_dict = {'linear':'No data transformation', 'sqrt':'Square-root transformation', 'log':'Log transformation'}

sim_type_label_dict = {'bdm': 'BDM', 'slm': 'SLM', 'demog': 'BDM'}

dataset_name_dict = {'david_et_al': 'David et al.', 'poyet_et_al': 'Poyet et al.', 'caporaso_et_al': 'Caporaso et al.'}


host_name_dict = {'david_et_al': {'DonorA_post_travel': 'A, post-travel', 'DonorA_pre_travel': 'A, pre-travel', 'DonorB_post_travel': 'B, post-travel', 'DonorB_pre_travel': 'B, pre-travel'}, 'poyet_et_al': {'ae': 'ae', 'am': 'am', 'an': 'an', 'ao': 'ao'}, 'caporaso_et_al': {'F4': 'F4', 'M3': 'M3'}}


def label_dataset_host(dataset, host):
    y_tick_label = '%s: %s' % (dataset_name_dict[dataset], host_name_dict[dataset][host])
    return y_tick_label


def make_blue_cmap(n):

    rgb_blue_phylo = cm.Blues(numpy.linspace(0,1,n+3))
    rgb_blue_phylo = mpl.colors.ListedColormap(rgb_blue_phylo[cmap_offset:,:-1])

    return rgb_blue_phylo




# https://github.com/weecology/macroecotools/blob/master/macroecotools/macroecotools.py
# code to cluster points
def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = numpy.log10(x), numpy.log10(y), numpy.log10(radius)
    for a, b in unique_points:
        if logscale == 1:
            loga, logb = numpy.log10(a), numpy.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 +
                                   (logy - logb) ** 2) <= logr ** 2])
        
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data






def plot_color_by_pt_dens(x, y, radius, loglog=0):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = numpy.array(sorted(plot_data, key=lambda point: point[2]))

    return sorted_plot_data





def get_scatter_density_arrays(x, y, loglog=0, radius=2):


    idx_to_keep = (x>0) & (y > 0)
    x = x[idx_to_keep]
    y = y[idx_to_keep]

    x_and_y = numpy.concatenate((x,y),axis=0)
    min_ = min(x_and_y)
    max_ = max(x_and_y)

    sorted_plot_data = plot_color_by_pt_dens(x, y, radius=radius, loglog=loglog)
    x,y,z = sorted_plot_data[:, 0], sorted_plot_data[:, 1], sorted_plot_data[:, 2]

    return x, y, z


def get_bin_log_x_mean_y(x, y, bins=20, min_n_in_bin=5):

    x_log10 = numpy.log10(x)
    #y_log10 = numpy.log10(y)
    y = numpy.asarray(y)

    hist_all, bin_edges_all = numpy.histogram(x_log10, density=True, bins=bins)
    #bins_x = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_x_to_keep = []
    bins_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        #y_log10_i = y_log10[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]
        y_i = y[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]

        if len(y_i) >= min_n_in_bin:
            bins_x_to_keep.append(bin_edges_all[i])
            bins_y.append(numpy.mean(y_i))


    bins_x_to_keep = numpy.asarray(bins_x_to_keep)
    bins_y = numpy.asarray(bins_y)

    bins_x_to_keep_no_nan = bins_x_to_keep[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]
    bins_y_no_nan = bins_y[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]
    
    # rescale
    bins_x_to_keep_no_nan = 10**bins_x_to_keep_no_nan

    return bins_x_to_keep_no_nan, bins_y_no_nan


def get_bin_x_mean_y(x, y, bins=20, min_n_in_bin=5):

    x = numpy.asarray(x)
    #y_log10 = numpy.log10(y)
    y = numpy.asarray(y)

    hist_all, bin_edges_all = numpy.histogram(x, density=True, bins=bins)
    #bins_x = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_x_to_keep = []
    bins_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        #y_log10_i = y_log10[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]
        y_i = y[(x>=bin_edges_all[i]) & (x<bin_edges_all[i+1])]

        if len(y_i) >= min_n_in_bin:
            bins_x_to_keep.append(bin_edges_all[i])
            bins_y.append(numpy.mean(y_i))


    bins_x_to_keep = numpy.asarray(bins_x_to_keep)
    bins_y = numpy.asarray(bins_y)

    bins_x_to_keep_no_nan = bins_x_to_keep[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]
    bins_y_no_nan = bins_y[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]
    
    # rescale
    #bins_x_to_keep_no_nan = 10**bins_x_to_keep_no_nan

    return bins_x_to_keep_no_nan, bins_y_no_nan




def get_bin_ci_log_x_mean_y(x, y, bins=20, min_n_in_bin=500, alpha=0.05):

    x_log10 = numpy.log10(x)
    #y_log10 = numpy.log10(y)
    y = numpy.asarray(y)

    hist_all, bin_edges_all = numpy.histogram(x_log10, density=True, bins=bins)
    #bins_x = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_x_to_keep = []
    lower_ci_y = []
    upper_ci_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        #y_log10_i = y_log10[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]
        y_i = y[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]

        if len(y_i) >= min_n_in_bin:
            bins_x_to_keep.append(bin_edges_all[i])
            y_i = numpy.sort(y_i)
            lower_ci_y.append(y_i[int(len(y_i)*(alpha/2))])
            upper_ci_y.append(y_i[int(len(y_i)*(1-(alpha/2)))])

            #bins_y.append(numpy.median(y_i))


    bins_x_to_keep = numpy.asarray(bins_x_to_keep)
    lower_ci_y = numpy.asarray(lower_ci_y)
    upper_ci_y = numpy.asarray(upper_ci_y)

    idx_to_keep = (~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(lower_ci_y)) & (~numpy.isnan(upper_ci_y))
    bins_x_to_keep_no_nan = bins_x_to_keep[idx_to_keep]

    lower_ci_y_no_nan = lower_ci_y[idx_to_keep]
    upper_ci_y_no_nan = upper_ci_y[idx_to_keep]

    # rescale
    bins_x_to_keep_no_nan = 10**bins_x_to_keep_no_nan

    return bins_x_to_keep_no_nan, lower_ci_y_no_nan, upper_ci_y_no_nan




# https://github.com/weecology/macroecotools/blob/master/macroecotools/macroecotools.py
# code to cluster points
def count_pts_within_radius(x, y, radius, logscale=(0,0)):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle

    # logscale is now a tuple to support semilog plots

    unique_points = set([(x[i], y[i]) for i in range(len(x))])
    count_data = []
    logx, logy, logr = numpy.log10(x), numpy.log10(y), numpy.log10(radius)
    for a, b in unique_points:
        if logscale == (1,1):
            loga, logb = numpy.log10(a), numpy.log10(b)
            num_neighbors = len(x[((logx - loga) ** 2 + (logy - logb) ** 2) <= logr ** 2])

        elif logscale == (1,0):
            loga = numpy.log10(a)
            num_neighbors = len(x[((logx - loga) ** 2 + (y - b) ** 2) <= logr ** 2])

        elif logscale == (0,1):
            logb = numpy.log10(b)
            num_neighbors = len(x[((x - a) ** 2 + (logy - logb) ** 2) <= logr ** 2])

        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])

        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = numpy.array(sorted(plot_data, key=lambda point: point[2]))

    return sorted_plot_data



def get_scatter_density_arrays(x, y, radius, loglog):


    if loglog == (1,1):
        idx_to_keep = (x>0) & (y>0)
        x = x[idx_to_keep]
        y = y[idx_to_keep]

    elif loglog == (1,0):
        idx_to_keep = (x>0)
        x = x[idx_to_keep]
        y = y[idx_to_keep]

    elif loglog == (0,1):
        idx_to_keep = (y>0)
        x = x[idx_to_keep]
        y = y[idx_to_keep]
    

    x_and_y = numpy.concatenate((x,y),axis=0)
    #min_ = min(x_and_y)
    #max_ = max(x_and_y)

    sorted_plot_data = plot_color_by_pt_dens(x, y, radius=radius, loglog=loglog)
    x,y,z = sorted_plot_data[:, 0], sorted_plot_data[:, 1], sorted_plot_data[:, 2]

    return x, y, z




def get_bin_mean_x_y(x, y, bins=20, min_n_bin=5):

    x_log10 = numpy.log10(x)
    y_log10 = numpy.log10(y)

    hist_all, bin_edges_all = numpy.histogram(x_log10, density=True, bins=bins)
    #bins_x = [0.5 * (bin_edges_all[i] + bin_edges_all[i+1]) for i in range(0, len(bin_edges_all)-1 )]
    bins_x_to_keep = []
    bins_y = []
    for i in range(0, len(bin_edges_all)-1 ):
        y_log10_i = y_log10[(x_log10>=bin_edges_all[i]) & (x_log10<bin_edges_all[i+1])]

        if len(y_log10_i) >= min_n_bin:
            bins_x_to_keep.append(bin_edges_all[i])
            bins_y.append(numpy.median(y_log10_i))


    bins_x_to_keep = numpy.asarray(bins_x_to_keep)
    bins_y = numpy.asarray(bins_y)

    bins_x_to_keep_no_nan = bins_x_to_keep[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]
    bins_y_no_nan = bins_y[(~numpy.isnan(bins_x_to_keep)) & (~numpy.isnan(bins_y))]

    bins_x_to_keep_no_nan = 10**bins_x_to_keep_no_nan
    bins_y_no_nan = 10**bins_y_no_nan

    return bins_x_to_keep_no_nan, bins_y_no_nan
