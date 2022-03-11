# 2-dimensional bGPFADS (Test Only)

The code implements the combination of (purely spherical) GPFADS (Rutten et al., 2020) and bGPFA (Jensen & Kao et al.,2021) but only within 2 dimensions. The code is based on bGPFA code from the paper (Jensen & Kao et al.,2021). To test bGPFADS, please use 1.py in mgplvm-pytorch or use the the jupyter notebook bGPFA_example.ipynb but set d_true and d_fit = 2.

###Below is the original README from bGPFA code

This directory contains the code accompanying ***Bayesian GPFA with automatic relevance determination and discrete noise models***.\
The code is divided into three parts: _mgplvm-pytorch/_ which contains the python package implementing bGPFA, _analysis/_ which contains the code used to fit the model and analyze the data in the paper, and _figures/_ which contains the code used to generate the figures in the paper.

In this readme, we provide a brief overview of each of these three parts as well as highlighting the key files used for different parts of the analysis.
Additionally, we have included a jupyter notebook _bGPFA_example.ipynb_ which generates and fits a small synthetic dataset to illustrate the method in a simple setting.

### mgplvm-pytorch/
This directory contains the actual python package implementing Bayesian GPFA.
To install, move to the _mgplvm-pytorch/_ directory and run the following commands:\
`conda create -n mgplvm python=3.8`\
`conda activate mgplvm`\
Install pytorch 1.7 with GPU support for the relevant CUDA version according to the pytorch documentation.\
`pip install .`\
Note that the code will in general work with different versions of python/pytorch and that the versions listed here are simply those used for our own analyses which we know not to have any incompatibility issues.\
To run the example notebook, an ipython kernel should be installed as well: `ipython kernel install --user --name "mgplvm"`.

***mgplvm/rdist/GP_base.py***\
This file contains the base implementation for our variational posterior over latents q(X).
_GP_circ.py_ is specific to the circulant parameterization used in our paper.
Note that the comparison of different parameterizations in Figure 8 was carried out in a separate Ocaml implementation.

***mgplvm/models/lgplvm.py***\
This file contains implementations of linear GPLVMs without (Lgplvm) and with (Lvgplvm) a variational q(F).
These are superclasses of the Gplvm class in _gplvm.py_.

***mgplvm/models/bfa.py***\
This file contains classes implementing the linear observation models p(F|X).
The variational Bvfa class was used for all our simulations since it facilitates non-Gaussian noise models, but in the Gaussian case it could be substituted for either the Bayesian non-variational Bfa class or the non-Bayesian non-variational Fa class.

***mgplvm/likelihood.py***\
This file contains the different noise models used in the paper.
In particular, we have used the Gaussian, Poisson and NegativeBinomial models with default settings.


### analysis/
This directory contains the code used for our analyses together with pre-trained models.
All analyses will automatically run on GPU if one is available -- they might take a while to run on CPU.
Note that pre-trained models and data are included so any script can be run without first training or processing the models.
However, some of these may require the availability of a GPU to load the pre-computed models.
To re-run all analyses in the paper, first run _synthetic_fit.py_ and _example_fit.py_, and run _fit_primate_data.py_ for both M1, S1 and M1+S1 with and without a 100 ms shift of the M1 spike times for 10 repetitions.
Then run _decode_kinematics.py_ for all models and _compute_repeats.py_.
Finally run the _plot_xxx.py_ scripts.
Figures can then be plotted using the code in _./figures/_ for the newly generated data.
Note that all figures can also be generated from the pre-trained models and pre-computed analyses which does not require installation of the mgplvm package.

***synthetic_fit.py***\
Fits synthetic LL and MSE data (figure 2a, 2b).\
use: `python synthetic_fit.py (true/false)`\
The optional command line argument specifies whether the script is run in crossvalidation mode or to compute training likelihoods (default).

***example_fit.py***\
Fits the synthetic examples with different noise models analyzed in figure 2c-e.\
use: `python example_fit.py`\
See also _bGPFA_example.ipynb_ for a simple model with negative binomial noise fitted to synthetic data.

***fit_primate_data.py***\
Fits bGPFA models to either the S1 data, M1 data, or the combined data for the analyses of a primate reaching task.\
use: `python fit_primate_data.py M1/S1/both true/false (\_rep*)`\
The first command line argument specifies which region to analyze.
The second command line argument specifies whether to shift the M1 spike times by 100 ms (true) or not (false).
The third optional command line argument takes the form \__rep3_ and will generate a new seed and save the model with a different name.
For the analyses in the paper, we used the base version (no third argument) as well as \__rep1_ to \__rep10_.
Running this file can be fairly memory intensive and should be done on a GPU with at least 12gb RAM.
Alternatively, the batch size can be reduced.

***decode_kinematics.py***\
Fits decoding models to either the S1, M1 or combined primate data.\
use: `python decode_kinematics.py M1/S1/both true/false (\_rep*)`\
Command line arguments are the same as in _fit_primate_data.py_.

***compute_repeats.py***\
Computes the ELBOs, number of retained dimensions, and decoding performance across 10 repeats with and without a 100 ms shift of the M1 data.\
use: `python compute_repeats.py`

***plot_xxx.py***\
Used for preliminary plots and simple analyses.
Writes data to _figures/figure_data/_ where the summary data is used to create the figures in the paper (note that this data is already included and figures can be plotted directly).

### figures/
Code used to generate the figures in the paper.

***plot_schematic.py --> figure 1***

***plot_synthetic.py --> figure 2***

***plot_primate.py --> figure 3***

***plot_primate_supp.py --> figure 4 (Appendix A)***

***plot_RT_supp.py --> figure 5 (Appendix B)***

***plot_quiescent.py --> figure 6 (Appendix C)***

***plot_dim_supp.py --> figure 7 (Appendix D)***
