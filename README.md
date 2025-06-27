[![DOI:10.1101/2023.03.02.530804](http://img.shields.io/badge/DOI-10.1101/2023.03.02.530804-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/2023.03.02.530804v1)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15754231.svg)](https://doi.org/10.5281/zenodo.15754231)


# sojourn_macroeco


Repository for code associated with the preprint: "The macroecological dynamics of sojourn trajectories in the human gut microbiome"





### Setting up your environment

You should be able to create a conda environment using a `environment.yml` file.

```bash
conda env create -f environment.yml
```


Alternatively, the code is written in Python 3.8 and requires the following packages: scipy, numpy, matplotlib, and statsmodels.




### Processing the data

```bash
R ~/GitHub/macroeco_phylo/scripts/dada2/dada2_caporaso_gut.T.py
R ~/GitHub/macroeco_phylo/scripts/dada2/dada2_david_gut.T.py
R ~/GitHub/macroeco_phylo/scripts/dada2/dada2_poyet_gut.T.py
```


### Running the analyses

```bash
sh ~/GitHub/macroeco_phylo/scripts/run_everything.sh
```

