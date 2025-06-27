#!/bin/bash

conda activate Py38





# Fig. 1

python ~/GitHub/sojourn_macroeco/scripts/plot_example_asv_trajectories.py

python ~/GitHub/sojourn_macroeco/scripts/plot_sojourn_time_data_mix.py
python ~/GitHub/sojourn_macroeco/scripts/plot_sojourn_time_vs_integral.py
python ~/GitHub/sojourn_macroeco/scripts/plot_sojourn_trajectory_data.py

# Fig. 2 (and Fig. SX)

python ~/GitHub/sojourn_macroeco/scripts/plot_fig2.py



# Supplemental figs
python ~/GitHub/sojourn_macroeco/scripts/plot_afd.py
python ~/GitHub/sojourn_macroeco/scripts/plot_timeseries.py
python ~/GitHub/sojourn_macroeco/scripts/plot_sojourn_dist_data_null.py