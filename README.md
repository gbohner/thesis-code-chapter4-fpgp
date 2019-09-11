# Learning interpretable models of latent stochastic dynamical systems

Code for Gergo Bohner's thesis work, Chapter 4. Full text: https://github.com/gbohner/thesis-pdf-git

--

This repository is a snapshot taken at thesis submission (14 June 2019) of the original repository https://github.com/gbohner/gpdm-fixpoints. It furthermore contains a more detailed README, explaining the use of individual code pieces to reproduce the figures in the thesis.

### Technical notes

This repository contains a mixture of Matlab, Python and latex code.

The Matlab version used to run the code was R2016b. For python I used an anaconda environment that may be reproduced via the ```environment.yml``` file, on Ubuntu 16.04 LTS. The latex code was used for visualisation, and should be standalone.

### Running the pipeline

The Matlab code concerns data simulation, and largely based on previously published code, as explained in the README within the folder ```thesis_Machens_Brody_sim```.

The main body of code concerns the implementation of the proposed FP GP-ADF model, discussed in the thesis in section 4.2.2. This code is in pure numpy python, and the fitting relies on the excellent ```autograd``` package, that implements automative differentiation of (almost) arbitrary numpy code.

The full model as well as the model fitting and evaluation code is implemented in the ```GPDM_direct_fixedpoints.ipynb``` notebook. A more detailed implementation that showcases the kernel derivatives discussed in Appendix C of the thesis is found in the ```RBF_kernel_derivatives_and_expectations.ipynb``` notebook.

The application to simulated datasets is found in the following files:

1.) The one-dimensional simulation of double well map and pitchfork bifurcation (section 4.3.2 of the thesis) is a reproduction of Figure 2 of our rejected ICML2018 submission, with the paper found on arXiv (https://arxiv.org/abs/1807.01486) and the original code on Github (https://github.com/gbohner/gpdm-fixpoints-icml2018-ver/tree/master/Pure_numpy_autograd). The model fit was carried out via the ```Experiment_1d_bifurcation_x3_varyslope.ipynb``` notebook, the figure data created by ```Figure_2.ipynb``` and saved to ```Figures/CSV/```, and finally visualised by ```Figures/figure_2.tex```. This figure corresponds to figure 4.2 in the thesis.

2.) The simulation and model fitting of mutually inhibiting neural populations observed via calcium imaging (section 4.3.5) of the thesis is implemented in the notebook ```thesis_Experiment_2d_MachensBrody.ipynb```. The figure data created by ```Figure_MachensBrody_spiking_calcium.ipynb``` and saved to ```Figures/CSV/```, and finally visualised via two latex files, ```Figures/make_machens_simulation_figure.tex``` for figure 4.5 of the thesis, and ```Figures/make_machens_fit_figure.tex``` for figure 4.6.








### Obtaining code for FP GP-SDE

The code for the FP GP-SDE algorithm discussed in the thesis in section 4.2.3 was implemented by Lea Duncker (https://github.com/leaduncker); please reach out to her for obtaining the latest version of the implementation.