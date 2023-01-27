# -*- coding: utf-8 -*-
#################### CONTENTS OF THE FILE ####################
"""
This file generates FIG. 5 in the paper. The plot generated is stored under the 
'saved_plots/' folder. 

· In the 'ADJUSTABLE PARAMETERS' section we can change the values of the 
variables defined there. Below is an explanation of what each variable does.

    GENERAL PARAMETERS
    -----------------------------------
    save_plot --> Boolean, Whether to save the plot generated or not.
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.

· In the 'DATA PREPARATION' section we load the mesh and wave function info
(from 'plot_data/') to the RAM. 

· In the 'PLOTTING' section we generate the plot corresponding to FIG. 5 in the
paper. All the wave functions plotted have Nhid=100 hidden neurons.
"""

#################### IMPORTS ####################
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator, 
                               AutoLocator)

# My modules
import N3LO

#################### ADJUSTABLE PARAMETERS ####################
# General parameters
save_plot = True
net_archs = ['1sc', '2sc', '1sd', '2sd']

#################### DATA PREPARATION ####################
# Uniform mesh with 1000 points
q_test = []
with open('../../deuteron_data/q_uniform_1000_10000/mesh.dat', 'r') as f:
    for line in f.readlines():
        try:
            q = float(line.split(' ')[2])
            q_test.append(q)
        except:
            try:
                q = float(line.split(' ')[3])
                q_test.append(q)
            except:
                pass
    f.close()
q_test_exact = q_test
n_test = len(q_test_exact)
q_test_exact2 = [q**2 for q in q_test_exact]
         
num1 = 0 
num = 700

# Exact wave functions
print('\nGathering potential data...')
n3lo = N3LO.N3LO(f'../../deuteron_data/q_uniform_{n_test}_10000/2d_vNN_J1.dat',
                 f'../../deuteron_data/q_uniform_{n_test}_10000/wfk.dat')

psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)
psi_target_s2, psi_target_d2 = [], []
for psi_s, psi_d in zip(psi_target_s, psi_target_d):   
    psi_target_s2.append(psi_s**2)
    psi_target_d2.append(psi_d**2)
fi, fj = q_test_exact2[num1:num], psi_target_s2[num1:num]
psi_target_s_norm = integrate.simpson([i*j for i, j in zip(fi, fj)],
                                      q_test_exact[num1:num])
fj = psi_target_d2[num1:num]
psi_target_d_norm = integrate.simpson([i*j for i, j in zip(fi, fj)],
                                      q_test_exact[num1:num])
psi_target_norm = psi_target_s_norm + psi_target_d_norm
psi_target_s_normalized = psi_target_s/np.sqrt(psi_target_norm)* \
    (1/np.sqrt(4*np.pi))
psi_target_d_normalized = psi_target_d/np.sqrt(psi_target_norm)* \
    (1/np.sqrt(4*np.pi))
    
# New normalization (good)
targ_s = psi_target_s_normalized
targ_d = psi_target_d_normalized

j1 = integrate.simpson([i*(j**2) for i, j in zip(q_test_exact2[num1:num],
                                                 targ_s[num1:num])],
                       q_test_exact[num1:num])
j2 = integrate.simpson([i*(j**2) for i, j in zip(q_test_exact2[num1:num],
                                                 targ_d[num1:num])],
                       q_test_exact[num1:num])
psi_target_norm = j1 + j2
norm_exact = 4*np.pi*psi_target_norm

# My wave functions
dict_psi_s, dict_psi_d = dict(), dict()
print('\nGathering wave function data...')
for arch in net_archs:
    with open(f'plot_data/wf_{arch}.txt', 'r') as f:
        c = 0
        psi_s, psi_d = [], []
        for line in f.readlines():
            if len(line) > 1:
                s, d = float(line.split(' ')[0]), float(line.split(' ')[1])
                psi_s.append(s)
                psi_d.append(d)
            else:       
                ls = [i*(j**2) for i, j in zip(fi, psi_s[num1:num])]
                ld = [i*(j**2) for i, j in zip(fi, psi_d[num1:num])]
                wf_norm_s2 = integrate.simpson(ls, q_test_exact[num1:num])
                wf_norm_d2 = integrate.simpson(ld, q_test_exact[num1:num])
                wf_norm2 = wf_norm_s2 + wf_norm_d2
                
                psi_s_prime = np.array(psi_s)* \
                    np.sqrt(psi_target_s_norm/wf_norm_s2)*(1/np.sqrt(4*np.pi))
                psi_d_prime = np.array(psi_d)* \
                    np.sqrt(psi_target_d_norm/wf_norm_d2)*(1/np.sqrt(4*np.pi))
                    
                ls_prime = [i*(j**2) for i, j in zip(q_test_exact2[num1:num], 
                                                     psi_s_prime[num1:num])]
                ld_prime = [i*(j**2) for i, j in zip(q_test_exact2[num1:num], 
                                                     psi_d_prime[num1:num])]
                j3 = integrate.simpson(ls_prime, q_test_exact[num1:num])
                j4 = integrate.simpson(ld_prime, q_test_exact[num1:num])
                psi_target_norm = j3 + j4
                norm_mine = 4*np.pi*psi_target_norm
                
                psi_s_prime = psi_s_prime*np.sqrt(norm_exact/norm_mine)
                psi_d_prime = psi_d_prime*np.sqrt(norm_exact/norm_mine)
                
                dict_psi_s[f"{arch}_{c}"] = psi_s_prime
                dict_psi_d[f"{arch}_{c}"] = psi_d_prime
                psi_s, psi_d = [], []
                c += 1
        f.close()

#################### PLOTTING ####################
# plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
# plt.rc('text', usetex=True)
# plt.rcParams.update(plt.rcParamsDefault)
# plt.ion()

# q_test = np.arange(0, 5, 5/1000)
q_test = q_test_exact

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 5), sharey='row',
                         sharex='row')
fig.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.25)
ax_psi_s = axes[0]
ax_psi_d = axes[1]

line_color_s = 'purple'
line_color_d = 'green'
linewidth = 0.15
linewidth_benchmark = 1.0
linewidth_benchmark_inset = 1.3

def inset_axes_s(ax):
    global axins
    axins = inset_axes(ax, 1, 1, loc='upper right', 
                       bbox_to_anchor=(0.9, 0.98),
                       bbox_transform=ax.transAxes) # position of the zoom box
    axins.set_xlim(0., 0.05)
    axins.set_ylim(3.2, 3.9)
    axins.tick_params(axis='both',labelsize=14)
    axins.yaxis.set_minor_locator(AutoMinorLocator())
    axins.xaxis.set_minor_locator(AutoMinorLocator())
    axins.set_xticks(np.arange(0., 0.101, 0.10))
    axins.set_yticks(np.arange(3.25, 3.7501, 0.5))
    axins.tick_params(axis='both', which='both', direction='in')
    axins.tick_params(axis='both', which='major', width=1.5)
    ax.indicate_inset_zoom(axins, edgecolor='black')
    
def inset_axes_d(ax):
    global axins
    axins = inset_axes(ax, 1, 1, loc='upper right', 
                       bbox_to_anchor=(0.98, 0.97),
                       bbox_transform=ax.transAxes) # position of the zoom box
    axins.set_xlim(0., 0.15)
    axins.set_ylim(0., 0.03)
    axins.yaxis.set_minor_locator(AutoMinorLocator())
    axins.xaxis.set_minor_locator(AutoMinorLocator())
    axins.tick_params(axis='both',labelsize=14)
    axins.tick_params(axis='both',which='minor',direction='in')
    ax.indicate_inset_zoom(axins,edgecolor='black')

#plt.rcParams.update(plt.rcParamsDefault)
#plt.ion()
#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
#plt.rc('text', usetex=True)

letters_S = ['(a)', '(b)', '(c)', '(d)']
letters_D = ['(e)', '(f)', '(g)', '(h)']
for arch in net_archs: 
    index = net_archs.index(arch)
    
    ### S-state ###
    ax = ax_psi_s[index]
    ax.set_ylim(0, 4)
    ax.set_xlim(0, 2)
    ax.set_title(f'{arch[0]} {arch[1:]}', fontsize=17)
    
    if index == 0:
        ax.set_ylabel(r"$\psi^{S}$", fontsize=23, rotation=0, labelpad=25)
    ax.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=5)
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    yticks = ax.yaxis.get_major_ticks()
    
    inset_axes_s(ax)
    wf_s_norm = []
    for key in list(dict_psi_s.keys()):
        if key.split('_')[0] == arch:
            wf = dict_psi_s[key]
            ax.plot(q_test, wf, linewidth=linewidth, color=line_color_s)
            axins.plot(q_test, wf, linewidth=linewidth, color=line_color_s)      
    ax.plot(q_test_exact, targ_s, linewidth=linewidth_benchmark, color='red',
            linestyle='dashed')
    ax.annotate(text=letters_S[index], 
                 xy=(0.85, 0.11),
                 xycoords='axes fraction',
                 fontsize=17)
    axins.plot(q_test_exact, targ_s, linewidth=linewidth_benchmark_inset, 
               color='red', linestyle='dashed')

    ### D-state ###
    ax = ax_psi_d[index]
    if index == 0:
        ax.set_ylabel("$\psi^{D}$", fontsize=23, rotation=0, labelpad=25)
    ax.set_xlabel("$q$ (fm$^{-1}$)", fontsize=15)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0., 0.07)
    ax.set_xticks(np.arange(0., 4.001, 1.))
    #ax.set_yticks(np.arange(0.,0.30001,0.1))
    ax.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    yticks = ax.yaxis.get_major_ticks()
    
    inset_axes_d(ax)
    
    wf_d_norm = []
    for key in list(dict_psi_d.keys()):
        if key.split('_')[0] == arch:
            wf = dict_psi_d[key]
            ax.plot(q_test, wf, linewidth=linewidth, color=line_color_d)
            axins.plot(q_test, wf, linewidth=linewidth, color=line_color_d)     
    ax.plot(q_test_exact, targ_d, linewidth=linewidth_benchmark, color='red',
            linestyle='dashed')
    ax.annotate(text=letters_D[index], 
                 xy=(0.85, 0.11),
                 xycoords='axes fraction',
                 fontsize=17)
    axins.plot(q_test_exact, targ_d, linewidth=linewidth_benchmark_inset,
               color='red', linestyle='dashed')

if save_plot == True:
    path_of_plot = '../../saved_plots/wf_variance_wf.pdf'
    plt.savefig(path_of_plot, format='pdf', bbox_inches='tight')
    print(f'Figure saved properly in {path_of_plot}.')
plt.pause(0.001)

plt.show()