# -*- coding: utf-8 -*-
#################### ADJUSTABLE OF THE FILE ####################
"""
This file generates FIG. 6 in the paper. The plot generated is stored under the
'saved_plots/' folder.

· In section 'GENERAL PARAMETERS' we can change the values of the variables. 
Below is an explanation of what each variable does.

    GENERAL PARAMETERS
    -----------------------------------
    save_plot --> Boolean, Whether to save the plot generated or not.
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    q_test --> NumPy Array, Test set.
    hidden_neurons_plot --> List, Contains the numbers of hidden neurons that
                            appear in the plot.

· In section 'DATA PREPARATION' we load the information under 'plot_data/' to 
the RAM for later plotting.

· In section 'PLOTTING' we make use of the previous data to generate the plot
of FIG. 6 in the paper. 
"""

#################### IMPORTS ####################
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter, AutoMinorLocator

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent_parent = os.path.dirname(parent)
sys.path.append(parent_parent)

os.chdir(current)

# My modules
from modules.aux_functions import strtobool

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="var_plot_stdev.py",
    usage="python3 %(prog)s [options]",
    description="Generates a plot displaying the variance of different NN "
    "wave functions.",
    epilog="Author: J Rozalén Sarmiento",
)
parser.add_argument(
    "--save-plot",
    help="Whether to save the plot or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
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
args = parser.parse_args()

#################### ADJUSTABLE PARAMETERS ####################
# General parameters
save_plot = args.save_plot
net_archs = args.archs
q_test = np.arange(0, 5, 5/1000)
hidden_neurons_plot = [20, 40, 60, 80, 100]

#################### DATA PREPARATION ####################
hidden_neurons = [20, 30, 40, 60, 80, 100]
hidden_neurons_sep = [20, 32, 40, 60, 80, 100]
dict_sigma_s, dict_sigma_d = dict(), dict()

# Wave functions
for arch in net_archs:
    with open(f'plot_data/var_{arch}.txt', 'r') as f:
        c = 0
        sigma_s, sigma_d = [], []
        for line in f.readlines():
            if len(line) > 1:
                s, d = float(line.split(' ')[0]), float(line.split(' ')[1])
                sigma_s.append(s)
                sigma_d.append(d)
            else:
                if arch != '2sd':
                    nhid = hidden_neurons[c]
                else:
                    nhid = hidden_neurons_sep[c]
                dict_sigma_s[f'{arch}_{nhid}'] = sigma_s
                dict_sigma_d[f'{arch}_{nhid}'] = sigma_d
                sigma_s, sigma_d = [], []
                c += 1
        f.close()

#################### PLOTTING ####################
linestyles = [(0, (3, 1, 1, 1)), 'dotted', 'dashed', 'dashdot']
#colors = ['#fdc086','#ffff99','#386cb0','#7fc97f','#beaed4']
#colors = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6']
#colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']
#colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']

letters_S = ['(a)', '(b)', '(c)', '(d)']
letters_D = ['(e)', '(f)', '(g)', '(h)']

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 5), sharey='row',
                         sharex='row')
fig.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.25)
ax_psi_s = axes[0]
ax_psi_d = axes[1]
linewidth = 1.5

for arch in net_archs: 
    index = net_archs.index(arch)
    
    ### S-state ###
    ax = ax_psi_s[index]
    ax.set_title(f'{arch[0]} {arch[1:]}', fontsize=17)
    if index == 0:
        ax.set_ylabel("$\sigma^{S}$", fontsize=23, rotation=0, labelpad=25)
        ax.set_xlim(0., 2.)
        ax.set_ylim(1e-5, 3e-1)
        ax.set_yscale('logit')
        ax.set_xscale('linear')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    ax.set_yticks([1e-5, 1e-3, 1e-1])
    ax.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    ax.tick_params(axis='x', pad=8)
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, 
                                           subs = np.arange(1.0, 10.0) * 0.1, 
                                           numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.annotate(text=letters_S[index], 
                 xy=(0.85, 0.85),
                 xycoords='axes fraction',
                 fontsize=17)
    c = 0
    for key in list(dict_sigma_s.keys()):
        if (key.split('_')[0] == arch and 
            int(key.split('_')[1]) in hidden_neurons_plot):
            wf = dict_sigma_s[key]
            ax.plot(q_test, wf, linewidth=linewidth, 
                    linestyle=linestyles[c] if c!= 4 else 'solid', 
                    color=colors[c])
            c +=1
      
    ### D-state ###
    ax = ax_psi_d[index]
    ax.set_xlabel("$q$ (fm$^{-1}$)", fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax.tick_params(axis='x', pad=8)
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    if index == 0:
        ax.set_xlim(0., 4.)
        ax.set_ylim(1e-5, 3e-1)
        ax.set_yscale('logit')
        ax.set_xscale('linear')
        ax.set_yticks([1e-5, 1e-3, 1e-1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel("$\sigma^{D}$", fontsize=23, rotation=0, labelpad=25)
    
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, 
                                           subs = np.arange(1.0, 10.0) * 0.1, 
                                           numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.annotate(text=letters_D[index], 
                 xy=(0.85, 0.85),
                 xycoords='axes fraction',
                 fontsize=17)
    c = 0
    for key in list(dict_sigma_d.keys()):
        if (key.split('_')[0] == arch and
            int(key.split('_')[1]) in hidden_neurons_plot):
            wf = dict_sigma_d[key]
            ax.plot(q_test, wf, linewidth=linewidth, 
                    linestyle=linestyles[c] if c!= 4 else 'solid', 
                    color=colors[c],
                    label='$N_\mathrm{hid}$'+f' ={hidden_neurons_plot[c]}'
                    if arch == '2sd' else '')
            c += 1
            
fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=6,
           fancybox=True, shadow=True, fontsize=14)
    
if save_plot == True:
    path_of_plot = '../../saved_plots/wf_variance_var.pdf'
    plt.savefig(path_of_plot, format='pdf', bbox_inches='tight')
    print(f'Figure saved properly in {path_of_plot}.')
    
plt.pause(0.001)
plt.show()


