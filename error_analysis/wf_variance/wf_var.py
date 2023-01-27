# -*- coding: utf-8 -*-
################## CONTENTS OF THE FILE ##################
"""
This file computes wave function variance data which is later used by files
'var_plot_wf.py' and 'var_plot_stdev.py' to generate Figs. 5 and 6 in the paper
respectively. 

· In Section 'ADJUSTABLE PARAMETERS' we can change the values of the variables.
An extensive explanation is given below. 

    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    device --> 'cpu', 'cuda', Physical processor into which we load our model.
    show_plot --> Boolean, Whether to show the plot or not.
    save_plot --> Boolean, Saves the plot at the end of the training or not.
    
    DEFAULT HYPERPARAM VALUES
    -----------------------------------
    hidden_neurons --> List, Contains different numbers of hidden neurons.
    default_actfun --> String, Default activation function.
    default_optim --> String, Default optimizer.
    default_lr --> float, Default learning rate.
    default_alpha --> float, Default smoothing constant.
    default_momentum --> float, Default momentum.

· In section 'DATA PREPARATION' we compute the wave functions for all four 
architectures and all hidden neuron numbers and store this in text files named
f'{arch}/wf_variance_nhid{nhid}_ntest{ntest}.txt'. 

· In section 'PREPARING DATA FOR PLOTTING' we compute and store the necessary
information for later plotting. This information is stored under 'plot_data/'.

· In section 'PLOTTING' we generate a plot with the wave function and variance
information for each architecture (i.e., four different plots). This is not
to be confused with Figs. 5 and 6 in the paper; these plots are an intermediate
step. 
"""

################## IMPORTS ##################
import torch, math, os, statistics, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from matplotlib.ticker import (FormatStrFormatter, AutoMinorLocator)
from tqdm import tqdm
sys.path.append('../..')

# My modules
from modules.aux_functions import dir_support 
import modules.integration as integration
import modules.neural_networks as neural_networks

################## ADJUSTABLE PARAMETERS ##################
# General parameters
net_archs = ['1sc', '2sc', '1sd', '2sd']
device = 'cpu'
show_plot = True
save_plot = True
nhid_for_plot = 100

# Default hyperparam values
hidden_neurons = [20, 30, 40, 60, 80, 100]
optim = 'RMSprop'
actfun = 'Sigmoid'
lr = 0.01
alpha = 0.9
mom = 0.0
epsilon = 1e-8

################## DATA PREPARATION ##################
# This number is taken from the computations carried out in 'var_plot_wf.py'
normalizing_factor = 1.1178590760902385/np.sqrt(4*np.pi) 
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
test_a = q_test[0]
test_b = q_test[-1]
n_test = len(q_test)
q_test = torch.tensor(q_test)
dir_support(['plot_data'])

x, w = np.polynomial.legendre.leggauss(64)
x_i = [torch.tensor(float(e)) for e in x]
w_i = [torch.tensor(float(e)) for e in w]

net_arch_map = {'1sc':neural_networks.sc_1,
                '2sc':neural_networks.sc_2,
                '1sd':neural_networks.sd_1,
                '2sd':neural_networks.sd_2}

dict_dict_of_seeds = {arch: {} for arch, _ in zip(net_archs, 
                                                  range(len(net_archs)))}
for arch in net_archs:
    print(f'\nComputing wave functions for ANN: {arch}...')
    for nhid in tqdm(hidden_neurons): 
        nhid = 32 if arch == '2sd' and nhid == 30 else nhid
        path_to_trained_models = f'../../saved_models/n3lo/{arch}/nhid{nhid}/'\
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/'
        path_to_filtered_runs = f'../error_data/{arch}/nhid{nhid}/' \
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/filtered_runs/'
        list_of_trained_models = os.listdir(path_to_trained_models)
        path_to_output_files = f'{arch}/'
        dir_support([path_to_output_files])
        
        # We search the models that have passed our filter and identify them
        file_of_filtered_runs = f'{path_to_filtered_runs}means_and_errors.txt'
        seeds_of_filtered_runs = []
        with open(file_of_filtered_runs, 'r') as file:
            for line in file.readlines():
                seed_i = int(line.split(' ')[12])
                seeds_of_filtered_runs.append(seed_i)  
            dict_dict_of_seeds[arch][nhid] = seeds_of_filtered_runs
                 
        # In the first iteration we remove all previously created files
        if arch == '1sc' and nhid == 20:
            for file in os.listdir(path_to_output_files):
                os.remove(f'{path_to_output_files}/{file}')
                #print(f'File {file} removed.')
                   
        ########## COMPUTING THE WAVE FUNCTIONS ##########
        seed_counter = 0
        for model in list_of_trained_models:
            seed = int(model.split('_')[0].replace('seed', ''))
            if seed not in dict_dict_of_seeds[arch][nhid]:   
                continue
            path_to_trained_model = f'{path_to_trained_models}{model}'
            output_file = f'{path_to_output_files}/wf_variance_nhid{nhid}'\
                f'_ntest{n_test}.txt'
            output_filename = output_file.split('/')[-1]
    
            # If the file we were going to create already exists, we erase  
            # the old version
            if seed_counter == 0:
                if os.path.isfile(output_file):
                    os.remove(output_file)
                
            # Single-layer specific params
            # ANN Parameters
            if arch == '1sc':
                Nhid_prime = nhid
            elif arch == '2sc' or arch == '1sd':
                Nhid_prime = int(nhid/2)
            elif arch == '2sd':
                Nhid_prime = int(nhid/4)
                
            Nin = 1
            Nout = 1 if arch == '2sd' else 2 
            W1 = torch.rand(Nhid_prime, Nin, requires_grad=True)*(-1.) 
            B = torch.rand(Nhid_prime)*2.-torch.tensor(1.) 
            W2 = torch.rand(Nout, Nhid_prime, requires_grad=True) 
            Ws2 = torch.rand(1, Nhid_prime, requires_grad=True) 
            Wd2 = torch.rand(1, Nhid_prime, requires_grad=True) 
            
            # We load our psi_ann to the CPU (or GPU)
            psi_ann = net_arch_map[arch](Nin, Nhid_prime, Nout, W1, 
                                         Ws2, B, W2, Wd2, actfun).to(device)
            psi_ann.load_state_dict(torch.load(f'{path_to_trained_models}' \
                                               f'{model}')['model_state_dict'])           
            psi_ann.eval()
            
            psi_s_pred = []
            psi_d_pred = []
            def test_loop(test_set, model):
                """Fills the lists 'psi_s_pred', 'psi_d_pred' with the 
                predicted values"""
                with torch.no_grad():
                    # Current wavefunction normalization 
                    # Note: we use the slow integrator because it needs not be 
                    # passed a specific mesh 
                    norm_s2 = lambda q : (q**2)*(model(q)[0])**2
                    norm_d2 = lambda q : (q**2)*(model(q)[1])**2
                    norm_s_ann = integration.gl64_function(norm_s2, test_a, 
                                                           test_b, x_i, w_i)
                    norm_d_ann = integration.gl64_function(norm_d2, test_a, 
                                                           test_b, x_i, w_i)
                    sqrt_norm_s_ann = torch.sqrt(norm_s_ann)
                    sqrt_norm_d_ann = torch.sqrt(norm_d_ann)
            
                    # We check that both square-rooted norms are non-zero
                    if sqrt_norm_s_ann > 0. and sqrt_norm_d_ann > 0.:
                        for x in test_set:
                            if arch != '2sc':
                                pred_s = model(x.unsqueeze(0).unsqueeze(0))[0]
                                pred_d = model(x.unsqueeze(0).unsqueeze(0))[1]
                            else:
                                pred_s = model(x.unsqueeze(0))[0]
                                pred_d = model(x.unsqueeze(0))[1]
                            # We store the values, skipping 'nans'
                            if (math.isnan(pred_s) == False and 
                                math.isnan(pred_d) == False):
                                psi_s_pred.append((pred_s/
                                                   sqrt_norm_s_ann).item())
                                psi_d_pred.append((pred_d/
                                                   sqrt_norm_d_ann).item())
                            else:
                                break
                        
                        if len(psi_s_pred) > 0 and len(psi_d_pred) > 0:
                            with open(output_file, 'a') as file:
                                for s,d in zip(psi_s_pred, psi_d_pred):
                                    file.write(f'{s*normalizing_factor} ' \
                                               f'{d*normalizing_factor}\n')
                                file.write('\n')
            test_loop(q_test, psi_ann)
            seed_counter += 1
        

    ################## PREPARING DATA FOR PLOTTING ##################    
    # Dictionaries that will store all the data
    # Key = Nhid, Value = list with the 20 initialisations
    the_big_boy_s = {}
    the_big_boy_d = {}
    the_big_boy_stdev_s = {}
    the_big_boy_stdev_d = {}
    
    path_to_output_files = f'{arch}/'
    
    # We read from the output file
    for output_file in os.listdir(path_to_output_files):
        Nhid = int(output_file.split('_')[2].split('.')[0].replace('nhid', ''))
        
        # We store the wave functions
        with open(f'{path_to_output_files}/{output_file}', 'r') as file:
            all_psi_s = []
            all_psi_d = []
            psi_s = []
            psi_d = []
            for line in file.readlines():
                if len(line.split(' ')) == 2 :
                    psi_s.append(float(line.split(' ')[0]))
                    psi_d.append(float(line.split(' ')[1]))
                else:
                    if (len(all_psi_s) < len(seeds_of_filtered_runs) and 
                        len(all_psi_d) < len(seeds_of_filtered_runs)):
                        all_psi_s.append(psi_s)
                        all_psi_d.append(psi_d)
                        psi_s = []
                        psi_d = []
        the_big_boy_s[Nhid] = all_psi_s
        the_big_boy_d[Nhid] = all_psi_d
                             
        # We compute the stdev
        stdev_vector_s = []
        stdev_vector_d = []
        c = 0
        while c < len(q_test):
            stdev_i_s = []
            stdev_i_d = []
            for wf_s, wf_d in zip(all_psi_s, all_psi_d):
                stdev_i_s.append(abs(wf_s[c]))
                stdev_i_d.append(abs(wf_d[c]))
            stdev_vector_s.append(statistics.stdev((stdev_i_s)))
            stdev_vector_d.append(statistics.stdev((stdev_i_d)))
            c += 1  
        the_big_boy_stdev_s[Nhid] = stdev_vector_s
        the_big_boy_stdev_d[Nhid] = stdev_vector_d
    
    # Write to file
    print('\n\nWriting data for plots...')
    with open(f'plot_data/wf_{arch}.txt', 'w') as f:
        # Format:
            # psi_s_seed0 psi_d_seed0
            # psi_s_seed1 psi_d_seed1 
            # ...
            # psi_s_seed20 psi_d_seed20
        for psi_s, psi_d in zip(the_big_boy_s[nhid_for_plot], 
                                the_big_boy_d[nhid_for_plot]):
            for psi_s_q, psi_d_q in zip(psi_s, psi_d):
                f.write(f'{psi_s_q} {psi_d_q}')
                f.write('\n')
            f.write('\n')
        f.close()
    with open(f'plot_data/var_{arch}.txt', 'w') as f:
        # Format:
            # sigma_s_nhid20 sigma_d_nhid20
            # sigma_s_nhid30 sigma_d_nhid30
            # ... 
            # sigma_s_nhid100 sigma_d_nhid100
        for nhid in hidden_neurons:
            nhid = 32 if arch == '2sd' and nhid == 30 else nhid
            for sigma_s, sigma_d in zip(the_big_boy_stdev_s[nhid],
                                        the_big_boy_stdev_d[nhid]):
                f.write(f'{sigma_s} {sigma_d}')
                f.write('\n')
            f.write('\n')
        f.close()

    ############################## PLOTTING ##############################
    # We generate the plot
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    plt.subplots_adjust(wspace=0.28, hspace=0.09)
    
    # S-state wf
    ax0 = axs[0,0]
    ax0.set_ylabel("$\psi^{L}$", fontsize=15)
    ax0.set_ylim(0, 4)
    ax0.set_xlim(0, 2)
    ax0.set_xticks(np.arange(0.0, 2.001, 0.5))
    ax0.tick_params(axis='x', labelcolor='white')
    ax0.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax0.tick_params(axis='both', which='major', width=1.5)
    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    # Zoomed region
    axins = zoomed_inset_axes(ax0, 2, loc='upper right', 
                              bbox_to_anchor=(0.8, 0.95),
                              bbox_transform=ax0.transAxes)
    axins.set_xlim(0., 0.05)
    axins.set_ylim(3.2, 3.9)
    #axins.set_xticks(np.arange(0.,0.101,0.05))
    axins.tick_params(axis='both',labelsize=14)
    axins.yaxis.set_minor_locator(AutoMinorLocator())
    axins.xaxis.set_minor_locator(AutoMinorLocator())
    axins.set_xticks(np.arange(0., 0.0401, 0.02))
    axins.tick_params(axis='both', which='minor', direction='in')
    axins.tick_params(axis='both', which='major',width=1.5)
    ax0.indicate_inset_zoom(axins, edgecolor='black')
    for wf in the_big_boy_s[nhid_for_plot]:            
        ax0.plot(q_test, wf, linewidth=0.3)
        axins.plot(q_test, wf)
    ax0.set_title('S-state, ($L=0$)', fontsize=15)
    
    # D-state wf
    ax1 = axs[0, 1]
    ax1.set_xlim(0, 4)
    #ax1.set_ylim(0.,1.25)
    ax1.set_xticks(np.arange(0., 4.001, 1.))
    #¡ax1.set_yticks(np.arange(0.,1.20001,0.4))  
    ax1.tick_params(axis='x', labelcolor='white')
    ax1.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax1.tick_params(axis='both', which='major', width=1.5)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Zoomed region
    axins = zoomed_inset_axes(ax1, 2.5, loc='upper right', 
                              bbox_to_anchor=(0.92, 0.97),
                              bbox_transform=ax1.transAxes)
    axins.set_xlim(0., 0.15)
    axins.set_ylim(0., 0.4)
    axins.yaxis.set_minor_locator(AutoMinorLocator())
    axins.xaxis.set_minor_locator(AutoMinorLocator())
    #axins.set_xticks(np.arange(0.,0.101,0.05))
    axins.tick_params(axis='both', labelsize=14)
    axins.tick_params(axis='both', which='minor', direction='in')
    ax1.indicate_inset_zoom(axins, edgecolor='black')
    for wf in the_big_boy_d[nhid_for_plot]: 
        ax1.plot(q_test, wf, linewidth=0.3)
        axins.plot(q_test, wf)
    
    ax1.set_title('D-state, ($L=2$)',fontsize=15)
    
    # S-state stdev
    ax2 = axs[1, 0] 
    ax2.set_xlabel("$q\,(\mathrm{fm^{-1}})$", fontsize=15)
    ax2.set_ylabel("$\sigma(\psi^{L})$", fontsize=15)
    ax2.set_yscale('logit')
    ax2.set_xscale('linear')
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_xlim(0., 2.)
    ax2.set_xticks(np.arange(0.0, 2.001, 0.5))
    ax2.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax2.tick_params(axis='both', which='major', width=1.5)
    for nhid in hidden_neurons:
        nhid = 32 if arch == '2sd' and nhid == 30 else nhid
        stdev_vector = the_big_boy_stdev_s[nhid]
        ax2.plot(q_test, stdev_vector, label=f'nhid = {nhid}', linewidth=0.5)
    
    # D-state stdev
    ax3 = axs[1, 1]
    ax3.set_xlabel("$q\,(\mathrm{fm^{-1}})$", fontsize=15)
    ax3.set_yscale('logit')
    ax3.set_xscale('linear')
    ax3.set_xlim(0., 4.)
    ax3.set_xticks(np.arange(0., 4.001, 1.))
    ax3.tick_params(axis='both', which='both', labelsize=15, direction='in')
    ax3.tick_params(axis='both', which='major',width=1.5)
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for nhid in hidden_neurons:
        nhid = 32 if arch == '2sd' and nhid == 30 else nhid
        stdev_vector = the_big_boy_stdev_d[nhid]
        ax3.plot(q_test, stdev_vector, 
                 label = '$N_\mathrm{hid}$'+' ={}'.format(nhid),
                 linewidth=0.5)
        
    ax3.legend(loc='upper center', bbox_to_anchor=(-0.25, -0.3), ncol=4,
               fancybox=True, fontsize=14)
    fig.suptitle(f'{arch}', fontsize=15)
    
    if save_plot == True:
        path_of_plot = f'plots/wf_variance_{arch}.pdf'
        plt.savefig(path_of_plot, format='pdf', bbox_inches='tight')
        print("Figure saved properly in {}.".format(path_of_plot))
    if show_plot == True:
        plt.pause(0.001)
        plt.show()