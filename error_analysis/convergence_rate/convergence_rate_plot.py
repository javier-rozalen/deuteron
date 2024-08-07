#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#################### CONTENTS OF THE FILE ####################
"""
In this file we generate the convergence rate plot corresponding to FIG. 2 in
the paper. The plot is saved under the 'saved_plots/' folder.

· In Section ADJUSTABLE PARAMETERS we can set the values of the variables 
according to our preferences. Below is an explanation of what each variable 
does.

    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    save_plot --> Boolean, Whether to save the plot or not.
    path_of_plot --> Str, The path where the plot will be stored.
                
    TRAINING HYPERPARAMETERS
    -----------------------------------
    hidden_neurons --> List, Contains different numbers of hidden neurons.
    actfuns --> List, Contains PyTorch-supported activation functions. 
    optimizers --> List, Contains PyTorch-supported optimizers.
    learning_rates --> List, Contains learning rates which must be specified 
                    in decimal notation.
    epsilon --> float, It appears in RMSProp and other optimizers.
    smoothing_constant --> List, It appears in RMSProp and other optimizers.
    momentum --> List, Contains values for the momentum of the optimizer.
    
    DEFAULT HYPERPARAM VALUES
    -----------------------------------
    default_actfun --> String, Default activation function.
    default_optim --> String, Default optimizer.
    default_lr --> float, Default learning rate.
    default_alpha --> float, Default smoothing constant.
    default_momentum --> float, Default momentum.
    
· In Section 'PLOTTING' we generate the plot and store it if necessary.
"""
########################## IMPORTS ##########################
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
from itertools import product
import os
import sys
import argparse

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent_parent = os.path.dirname(parent)
sys.path.append(parent_parent)

current = os.path.dirname(os.path.realpath(__file__))
os.chdir(current)

# My modules
from modules.aux_functions import strtobool


##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="convergence_rate_plot.py",
    usage="python3 %(prog)s [options]",
    description="Generates the convergence rate plot.",
    epilog="Author: J Rozalén Sarmiento",
)
parser.add_argument(
    "--explore-hyperparams",
    help="Whether to train for the different combinations of hyperparameters "
    "specified or not (default: False). WARNING: this action will disable "
    "hyperparameter exploration even if lists of hyperparameters were given.",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--save-plot",
    help="Whether to save the plot or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--archs",
    help="List of NN architectures to train (default: 1sc 2sc 1sd 2sd)"
    "WARNING: changing this might entail further code changes to ensure proper"
    " functioning)",
    default=["1sc", "2sc", "1sd", "2sd"],
    nargs="*",
    type=str
)
parser.add_argument(
    "--hidden-nodes",
    help="List of hidden node numbers to use (default: 20 30 40 60 80 100)",
    default=[20, 30, 40, 60, 80, 100],
    nargs="*",
    type=int
)
parser.add_argument(
    "--activations",
    help="List of activation functions (default: Sigmoid Softplus ReLU)",
    default=["Sigmoid", "Softplus", "ReLU"],
    nargs="*",
)
parser.add_argument(
    "--optimizers",
    help="List of optimizers (default: RMSprop)",
    default=["RMSprop"],
    nargs="*",
)
parser.add_argument(
    "--lrs",
    help="List of learning rates (default: 0.005 0.01 0.05)",
    default=[0.005, 0.01, 0.05],
    nargs="*",
    type=float
)
parser.add_argument(
    "--alphas",
    help="List of smoothing constants (default: 0.7 0.8 0.9)",
    default=[0.7, 0.8, 0.9],
    nargs="*",
    type=float
)
parser.add_argument(
    "--mus",
    help="List of momentum values (default: 0.0 0.9)",
    default=[0.0, 0.9],
    nargs="*",
    type=float
)
args = parser.parse_args()


#################### ADJUSTABLE PARAMETERS ####################
# General parameters
net_archs = args.archs
save_plot = args.save_plot
path_of_plot = '../../saved_plots/convergence_rate.pdf'

# Variable hyperparameters
hidden_neurons = args.hidden_nodes
actfuns = args.activations
optimizers = args.optimizers
learning_rates = args.lrs 
smoothing_constant = args.alphas
momentum = args.mus

# Default values of hyperparameters
default_actfun = 'Sigmoid'
default_optim = 'RMSprop'
default_lr = 0.01
default_alpha = 0.9
default_momentum = 0.0

if not args.explore_hyperparams:
    actfuns = [default_actfun]
    optimizers = [default_optim]
    learning_rates = [default_lr]
    smoothing_constant = [default_alpha]
    momentum = [default_momentum]

########################## PLOTTING ##########################
fig, ax = plt.subplots()
fig.tight_layout()
ax.set_xlabel('$N_\mathrm{hid}$', fontsize=18)
ax.set_ylabel('Rate, $r$', fontsize=18)

ax.set_ylim(0., 1.05) 
ax.set_xlim(20, 100)
ax.tick_params(axis='both', which='both', direction='in', labelsize=15)
ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
ax.xaxis.set_major_locator(MultipleLocator(20))

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

c = 0
for arch in net_archs:
    if arch != '2sd':
        x_axis = hidden_neurons
    else:
        x_axis = [20, 32, 40, 60, 80, 100]
    y_axis = []
    for nhid, optim, actfun, lr, alpha, mom in product(x_axis, optimizers, 
                                                       actfuns, learning_rates, 
                                                       smoothing_constant, 
                                                       momentum): 
        
        # We restrict the search to a specific combination of hyperparameters, 
        # normally the default combination.
        if (optim == default_optim and
            actfun == default_actfun and
            lr == default_lr and
            alpha == default_alpha and
            mom == default_momentum):
            
                # We extract the data from the error file
                conv_rate_file = f'../error_data/{arch}/nhid{nhid}/' \
                    f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/' \
                    'conv_rate.txt'
                with open(conv_rate_file, 'r') as file:
                    cv = float(file.readlines()[0].split(' ')[2])
                    y_axis.append(cv)
                    file.close()
                
    # Plot
    ax.plot(x_axis, y_axis, linestyle=linestyles[c], 
            label=arch, linewidth=3.)
    c += 1
        
fig.legend(loc='lower center', bbox_to_anchor=(0.84, 0.34), ncol=1,
           fancybox=True, fontsize=13)

if save_plot:
    plt.savefig(path_of_plot, format='pdf', bbox_inches='tight')
    print('Figure saved in deuteron/saved_plots/convergence_rate.pdf')