# -*- coding: utf-8 -*-

############################## IMPORTANT INFORMATION #################################
"""
Under the 'FURTHER ERROR COMPUTATIONS' section of the code, we first go over all the error files 
generated with 'error_measure.py',and we compute further necessary errors. For each network architecture, 
we store all this information in a single file named 'joint_graph.txt', which contains all the necessary 
information to generate the joint graph. Then, the data of this file is appended to the 'master_plot_data_?.txt'
files, each of which contains data from one network architecture.

In the 'DATA PREPARATION FOR PLOTTING' section of the code we read the files created in the previous
section and load dictionaries with it. 

In the 'PLOTTING' section of the code the dictionaries in the previous section are read and the data
is finally plotted.

Parameters to adjust manually:
    GENERAL PARAMETERS
    save_plot --> Boolean, Sets whether the plot is saved or not.
    path_of_plot --> str, relative_directory/name_of_plot.pdf
    show_osc_error --> Boolean, (self-explanatory)
    show_stoch_error --> Boolean, (self-explanatory)
    variable_hyperparam_list --> list, Defines the hyperparameter to explore
    
INSTRUCTIONS OF USE

First set the GENERAL PARAMETERS to your desired values, and then go to the loop
'for vh in variable_hyperparam_list' and set the variable corresponding to the
variable hyperparameter to 'vh'.
"""

###################################### IMPORTS ######################################
import matplotlib.pyplot as plt
import numpy as np
import math, os, statistics, shutil, sys, pathlib
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import (MultipleLocator)

initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

from modules.aux_functions import round_to_1, dir_support, get_keys_from_value

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
net_archs = ['1sc', '2sc', '1sd', '2sd']
save_plot = False
show_osc_error = True
show_stoch_error = True
shades_or_bars = 'shades'

# Variable hyperparams
actfuns = ['Sigmoid', 'Softplus', 'ReLU']
learning_rates = ['0.005', '0.01', '0.05'] # Use decimal notation 
smoothing_constant = ['0.7', '0.8', '0.9']
momentum = ['0.0', '0.9']

variable_hyperparam_list = momentum

default_actfun = 'Sigmoid'
default_lr = '0.01'
default_alpha = '0.9'
default_momentum = '0.0'

###################################### MISC. ######################################
folders_map = {'actfuns': actfuns,
               'learning_rates': learning_rates,
               'alpha': smoothing_constant,
               'mom': momentum}
variable_hyperparam_name = get_keys_from_value(d=folders_map, 
                                               val=variable_hyperparam_list)[0]
path_of_plot = f'saved_plots/energy_{variable_hyperparam_name}_plot.pdf'

dir_support(['plotting_data', f'{variable_hyperparam_name}'])
dir_support(['saved_plots'])
dir_support(['variable_hyperparams', f'{variable_hyperparam_name}'])

plt.rcParams['xtick.major.pad'] = '9'

############################## FURTHER ERROR COMPUTATIONS #################################
for arch in net_archs: 
    """We iterate over all the network architectures."""
        
    print(f'\nArch: {arch}')
    path_to_error_files = f'variable_hyperparams/' \
        f'{variable_hyperparam_name}/'
        
    # We copy the different files corresponding to the different values of the
    # variable hyperparam into a folder for later analysis
    for vh in variable_hyperparam_list: 
        # We keep the rest of the hyperparameters fixed to their default values
        actfun = default_actfun
        lr = default_lr
        alpha = default_alpha
        mom = vh
    
        # We look for the (filtered) error files.
        old_path_to_error_files = f'../error_data/{arch}/' \
            f'{actfun}/lr{lr}/alpha{alpha}/mu{mom}/filtered_runs/'
        
        files = os.listdir(path=old_path_to_error_files)
        for file in files:
            original_filename = f'{old_path_to_error_files}{file}'
            new_filename = f'{path_to_error_files}{vh}_{arch}.txt'
            shutil.copyfile(src=original_filename, dst=new_filename)
   
    # joint_graph_file creation    
    joint_graph_file = f'{path_to_error_files}joint_graph_{arch}.txt'
    
    # We check wether 'joint_graph_file.txt' exists and we delete it if it does
    if os.path.isfile(joint_graph_file):
        os.remove(joint_graph_file)
        
    # We create the file 'master_plot_data_{arch}.txt'
    # under the 'plotting_data' folder. 
    with open(f'plotting_data/{variable_hyperparam_name}/master_plot_data_{arch}.txt', 'w') as global_file:
        global_file.close()
            
    all_files = os.listdir(path_to_error_files)
    list_of_error_files = []
    
    # We select the error files among all the files in this directory
    for file in all_files:
        if (file[-3:] == 'txt' 
            and file.split('_')[0] in variable_hyperparam_list 
            and file.split('_')[1].split('.')[0] == arch):
            list_of_error_files.append(file)
                   
    # We sort the list of error files so that the data will be written in the
    # order of the variable hyperparameter lists
    list_of_error_files = sorted(list_of_error_files, 
                                 key=lambda x : variable_hyperparam_list.index(x.split('_')[0]))
        
    # We retrieve the different values for E, ks, kd, PD (from different seeds)
    # and store them in lists. l
    print(f'List of error files: {list_of_error_files}')
    for file in list_of_error_files:
        print(f'Analyzing file: {file}')
        ################## ERROR DATA FETCHING ##################
        filename = '{}{}'.format(path_to_error_files, file) 
        
        # Nhid = int(file.split('.')[0].replace('nhid', ''))
        
        energies = []
        E_top_errors = []
        E_bot_errors = []
        E_errors = [E_bot_errors, E_top_errors]
        
        ks = []
        ks_top_errors = []
        ks_bot_errors = []
        
        kd = []
        kd_top_errors = []
        kd_bot_errors = []
        
        pd = []
        pd_top_errors = []
        pd_bot_errors = []
        
        # We fill the lists above with the parameters computed 
        # with error_measure.py and later filtered with filter.py
        with open(filename, 'r') as file2:
            for line in file2.readlines():
                energies.append(float(line.split(' ')[0]))
                E_top_errors.append(float(line.split(' ')[1]))
                E_bot_errors.append(float(line.split(' ')[2]))
                ks.append(float(line.split(' ')[3]))
                ks_top_errors.append(float(line.split(' ')[4]))
                ks_bot_errors.append(float(line.split(' ')[5]))
                kd.append(float(line.split(' ')[6]))
                kd_top_errors.append(float(line.split(' ')[7]))
                kd_bot_errors.append(float(line.split(' ')[8]))
                pd.append(float(line.split(' ')[9]))
                pd_top_errors.append(float(line.split(' ')[10]))
                pd_bot_errors.append(float(line.split(' ')[11]))
            file2.close()
        
        ################## FURTHER ERRORS COMPUTATION ##################  
        # We use the lists above to compute meaningful statistical quantities
        ks_errors = [ks_bot_errors, ks_top_errors]
        kd_errors = [kd_bot_errors, kd_top_errors]
        pd_errors = [pd_bot_errors, pd_top_errors]
        
        if len(energies) > 1:
            mean_E = sum(energies) / len(energies)
            stdev_E = statistics.stdev(energies)/math.sqrt(len(energies))
            mean_E_top_errors = sum(E_top_errors)/len(E_top_errors)
            mean_E_bot_errors = sum(E_bot_errors)/len(E_bot_errors)
            
            mean_ks = sum(ks) / len(ks)
            stdev_ks = statistics.stdev(ks)/math.sqrt(len(ks))
            mean_ks_top_errors = sum(ks_top_errors)/len(ks_top_errors)
            mean_ks_bot_errors = sum(ks_bot_errors)/len(ks_bot_errors)
            
            mean_kd = sum(kd) / len(kd)
            stdev_kd = statistics.stdev(kd)/math.sqrt(len(kd)) # Stochastic error (used for both upper and lower bounds)
            mean_kd_top_errors = sum(kd_top_errors)/len(kd_top_errors) # Oscillating error (upper bound)
            mean_kd_bot_errors = sum(kd_bot_errors)/len(kd_bot_errors) # Oscillating error (lower bound)
                    
            # This saves all the parameters necessary to compute 
            # the final graph to the file 'joint_graph_{arch}.txt'
            with open(joint_graph_file, 'a') as file3:
                # E, sigma(E), E+, E-, Ks, sigma(Ks), Ks+, Ks-, Pd, sigma(Pd), Pd+, Pd-
                file3.write(str(mean_E)+' '+
                    str(stdev_E)+' '+
                    str(mean_E_top_errors)+' '+
                    str(mean_E_bot_errors)+' '+
                    str(mean_ks)+' '+
                    str(stdev_ks)+' '+
                    str(mean_ks_top_errors)+' '+
                    str(mean_ks_bot_errors)+' '+
                    str(mean_kd)+' '+
                    str(stdev_kd)+' '+
                    str(mean_kd_top_errors)+' '+
                    str(mean_kd_bot_errors)+' \n')
                file3.close()
        else:
            print(f'energies has {len(energies)} elements, so we cannot' \
                  ' compute the statistics...')
    

    ################## FURTHER DATA FETCHING FOR PLOTTING ##################
    """We read the data of joint_graph_file, and compute the necessary stuff 
    for the fancy joint plot."""
    energies_joint = [] 
    mean_E_errors = []
    E_top_errors_joint = []
    E_bot_errors_joint = []
    
    ks_joint = []
    mean_ks_errors = []
    ks_top_errors_joint = []
    ks_bot_errors_joint = []
    
    kd_joint = []
    mean_kd_errors = []
    kd_top_errors_joint = []
    kd_bot_errors_joint = []
    
    # We read the data to fill the lists above
    with open(joint_graph_file, 'r') as file4:
        for line in file4.readlines():
            energies_joint.append(float(line.split(' ')[0]))
            mean_E_errors.append(float(line.split(' ')[1]))
            E_top_errors_joint.append(float(line.split(' ')[2]))
            E_bot_errors_joint.append(float(line.split(' ')[3]))
            ks_joint.append(float(line.split(' ')[4]))
            mean_ks_errors.append(float(line.split(' ')[5]))
            ks_top_errors_joint.append(float(line.split(' ')[6]))
            ks_bot_errors_joint.append(float(line.split(' ')[7]))
            kd_joint.append(float(line.split(' ')[8]))
            mean_kd_errors.append(float(line.split(' ')[9]))
            kd_top_errors_joint.append(float(line.split(' ')[10]))
            kd_bot_errors_joint.append(float(line.split(' ')[11]))
        file4.close()
    """
    # We generate lists with the complete error data.
    E_errors_joint = [E_bot_errors_joint,E_top_errors_joint]    
    ks_errors_joint = [ks_bot_errors_joint,ks_top_errors_joint]
    kd_errors_joint = [kd_bot_errors_joint,kd_top_errors_joint]
    """
    E_osc_shade_top = [energies_joint[i]+abs(E_top_errors_joint[i]) for i in range(len(energies_joint))]
    E_osc_shade_bot = [energies_joint[i]-abs(E_bot_errors_joint[i]) for i in range(len(energies_joint))]
    E_stoch_shade_top = [energies_joint[i]+abs(mean_E_errors[i]) for i in range(len(energies_joint))]
    E_stoch_shade_bot = [energies_joint[i]-abs(mean_E_errors[i]) for i in range(len(energies_joint))]
    
    ks_osc_shade_top = [ks_joint[i]+abs(ks_top_errors_joint[i]) for i in range(len(ks_joint))]
    ks_osc_shade_bot = [ks_joint[i]-abs(ks_bot_errors_joint[i]) for i in range(len(ks_joint))]
    ks_stoch_shade_top = [ks_joint[i]+abs(mean_ks_errors[i]) for i in range(len(ks_joint))]
    ks_stoch_shade_bot = [ks_joint[i]-abs(mean_ks_errors[i]) for i in range(len(ks_joint))]
    
    kd_osc_shade_top = [kd_joint[i]+abs(kd_top_errors_joint[i]) for i in range(len(kd_joint))]
    kd_osc_shade_bot = [kd_joint[i]-abs(kd_bot_errors_joint[i]) for i in range(len(kd_joint))]
    kd_stoch_shade_top = [kd_joint[i]+abs(mean_kd_errors[i]) for i in range(len(kd_joint))]
    kd_stoch_shade_bot = [kd_joint[i]-abs(mean_kd_errors[i]) for i in range(len(kd_joint))]
    
    # HAHAHA Xd
    print(f'E_osc_top = {[round_to_1(x) for x in E_top_errors_joint]}')
    print(f'E_osc_bot = {[round_to_1(x) for x in E_bot_errors_joint]}')
    print(f'E_stoch_top = {[round_to_1(x) for x in [abs(energies_joint[i] - E_stoch_shade_top[i]) for i in range(len(energies_joint))]]}')
    print(f'E_stoch_bot = {[round_to_1(x) for x in [abs(energies_joint[i] - E_stoch_shade_bot[i]) for i in range(len(energies_joint))]]}')
        
    # We save the data in a global file.
    with open(f'plotting_data/{variable_hyperparam_name}/master_plot_data_{arch}.txt','a') as global_file:
        
        for E,E_osc_bot,E_osc_top,E_stoch_bot,E_stoch_top,Fs,Fs_osc_bot,Fs_osc_top,Fs_stoch_bot,Fs_stoch_top,Fd,Fd_osc_bot,Fd_osc_top,Fd_stoch_bot,Fd_stoch_top in zip(energies_joint,E_osc_shade_bot,E_osc_shade_top,
                                                                                                          E_stoch_shade_bot,E_stoch_shade_top,
                                                                                                          ks_joint,ks_osc_shade_bot,ks_osc_shade_top,
                                                                                                          ks_stoch_shade_bot,ks_stoch_shade_top,
                                                                                                          kd_joint,kd_osc_shade_bot,kd_osc_shade_top,
                                                                                                          kd_stoch_shade_bot,kd_stoch_shade_top):
            global_file.write(f'{E} {E_osc_bot} {E_osc_top} {E_stoch_bot} {E_stoch_top} {Fs} {Fs_osc_bot} {Fs_osc_top} {Fs_stoch_bot} {Fs_stoch_top} {Fd} {Fd_osc_bot} {Fd_osc_top} {Fd_stoch_bot} {Fd_stoch_top}\n')
        global_file.close() 
            
    print(f'Error data of model {arch}_{variable_hyperparam_name} successfully' \
          f' appended to master_plot_data_{arch}.txt.')
        
################################# DATA PREPARATION FOR PLOTTING #################################
if True:
    print('\n')
    data_list = ['E', 'E_osc_bot', 'E_osc_top', 'E_stoch_bot', 'E_stoch_top', 
                 'Fs', 'Fs_osc_bot', 'Fs_osc_top', 'Fs_stoch_bot', 'Fs_stoch_top', 
                 'Fd', 'Fd_osc_bot', 'Fd_osc_top', 'Fd_stoch_bot', 'Fd_stoch_top']
    
    dict_1sc = {}
    dict_2sc = {}
    dict_1sd = {}
    dict_2sd = {}
    dict_E = {}
    dict_F = {}
    dict_dict = {'1sc':dict_1sc,
                 '2sc':dict_2sc,
                 '1sd':dict_1sd,
                 '2sd':dict_2sd}
    
    # We create the keys
    for e in data_list:
        key_1sc = f'1sc_{e}'
        key_2sc = f'2sc_{e}'
        key_1sd = f'1sd_{e}'
        key_2sd = f'2sd_{e}'
        dict_1sc[key_1sc] = list()
        dict_2sc[key_2sc] = list()
        dict_1sd[key_1sd] = list()
        dict_2sd[key_2sd] = list()
        
    for arch in list(net_archs):
        print(f'Loading dictionary of model {arch}...')
        with open(f'plotting_data/{variable_hyperparam_name}/master_plot_data_{arch}.txt', 'r') as f:
            list_of_keys = list(dict_dict[arch].keys())
            for line in f.readlines():
                line = line.split(' ')
                c = 0
                for item in line:
                    key_at_index = list_of_keys[c]
                    dict_dict[arch][key_at_index].append(float(item))
                    c += 1
            f.close()         
            
        # We fill a dictionary with the energies and another one with the fidelities
        for e in list(dict_dict[arch].keys()):
            if e.split('_')[1] == 'E':
                dict_E[e] = dict_dict[arch][e]
            else:
                dict_F[e] = dict_dict[arch][e]
            
################################# PLOTTING #################################
if True:
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 6), sharey='row',
                             sharex='col')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.1)
    ax_E = axes[0]
    ax_F = axes[1]
    x_axis = variable_hyperparam_list
    
    ### Energies ###
    # 1sc
    ax = ax_E[0]
    ax.set_title('1 sc', fontsize=17)
    ax.set_ylabel('E (Mev)', fontsize=15)
    ax.set_ylim(-2.227, -2.220)
    #ax.yaxis.set_ticklabels(['-2.227','-2.226','-2.225','-2.224','-2.223','-2.222','-2.221','-2.220'],
    #fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    
    ax.axhline(y=-2.2267, color="red", linestyle="--", label='Exact')
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error = abs(np.array(dict_E['1sc_E'])-np.array(dict_E['1sc_E_osc_bot']))
            upper_error = abs(np.array(dict_E['1sc_E'])-np.array(dict_E['1sc_E_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['1sc_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error = abs(np.array(dict_E['1sc_E'])-np.array(dict_E['1sc_E_stoch_bot']))
            upper_error = abs(np.array(dict_E['1sc_E'])-np.array(dict_E['1sc_E_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['1sc_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_E['1sc_E'], label='Energy')
        if show_osc_error:
            ax.fill_between(x_axis, dict_E['1sc_E_osc_bot'], dict_E['1sc_E_osc_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.3)
        if show_stoch_error:
            ax.fill_between(x_axis, dict_E['1sc_E_stoch_bot'], dict_E['1sc_E_stoch_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.5)
    
    # 2sc
    ax = ax_E[1]
    ax.set_title('2 sc', fontsize=17)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error = abs(np.array(dict_E['2sc_E'])-np.array(dict_E['2sc_E_osc_bot']))
            upper_error = abs(np.array(dict_E['2sc_E'])-np.array(dict_E['2sc_E_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['2sc_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error = abs(np.array(dict_E['2sc_E'])-np.array(dict_E['2sc_E_stoch_bot']))
            upper_error = abs(np.array(dict_E['2sc_E'])-np.array(dict_E['2sc_E_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['2sc_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_E['2sc_E'])
        if show_osc_error:
            ax.fill_between(x_axis, dict_E['2sc_E_osc_bot'], dict_E['2sc_E_osc_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.3)
        if show_stoch_error:
            ax.fill_between(x_axis, dict_E['2sc_E_stoch_bot'], dict_E['2sc_E_stoch_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.5)
    
    
    # 1sd
    ax = ax_E[2]
    ax.set_title('1 sd', fontsize=17)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error = abs(np.array(dict_E['1sd_E'])-np.array(dict_E['1sd_E_osc_bot']))
            upper_error = abs(np.array(dict_E['1sd_E'])-np.array(dict_E['1sd_E_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['1sd_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error = abs(np.array(dict_E['1sd_E'])-np.array(dict_E['1sd_E_stoch_bot']))
            upper_error = abs(np.array(dict_E['1sd_E'])-np.array(dict_E['1sd_E_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['1sd_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_E['1sd_E'])
        if show_osc_error:
            ax.fill_between(x_axis, dict_E['1sd_E_osc_bot'], dict_E['1sd_E_osc_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.3)
        if show_stoch_error:
            ax.fill_between(x_axis, dict_E['1sd_E_stoch_bot'], dict_E['1sd_E_stoch_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.5)
    
        
    # 2sd
    ax = ax_E[3]
    ax.set_title('2 sd', fontsize=17)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error = abs(np.array(dict_E['2sd_E'])-np.array(dict_E['2sd_E_osc_bot']))
            upper_error = abs(np.array(dict_E['2sd_E'])-np.array(dict_E['2sd_E_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['2sd_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error = abs(np.array(dict_E['2sd_E'])-np.array(dict_E['2sd_E_stoch_bot']))
            upper_error = abs(np.array(dict_E['2sd_E'])-np.array(dict_E['2sd_E_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_E['2sd_E'],
                        label='Energy',
                        yerr=[lower_error, upper_error],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_E['2sd_E'])
        if show_osc_error:
            ax.fill_between(x_axis, dict_E['2sd_E_osc_bot'], dict_E['2sd_E_osc_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.3)
        if show_stoch_error:
            ax.fill_between(x_axis, dict_E['2sd_E_stoch_bot'], dict_E['2sd_E_stoch_top'],
                            edgecolor='#089FFF', facecolor='#089FFF', alpha=0.5)
    
    
    ### Fidelities ###
    d_shades = {'stoch':'#249225',
                'osc':'#abd5ab',
                'alpha':0.5}
    
    # 1sc
    ax = ax_F[0]
    ax.set_ylabel('Fidelity', fontsize=15)
    ax.set_ylim(0.99992, 1.000005)
    #ax.yaxis.set_ticklabels(['0.99990','0.99992','0.99994','0.99996','0.99998','1.00000',],
    #fontsize=14)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
    #ax.set_xlabel(f'{variable_hyperparam_name}', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.00001))
    
    ax.axhline(y=1., color="red", linestyle="--")
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error_S = abs(np.array(dict_F['1sc_Fs'])-np.array(dict_F['1sc_Fs_osc_bot']))
            upper_error_S = abs(np.array(dict_F['1sc_Fs'])-np.array(dict_F['1sc_Fs_osc_top']))
            lower_error_D = abs(np.array(dict_F['1sc_Fd'])-np.array(dict_F['1sc_Fd_osc_bot']))
            upper_error_D = abs(np.array(dict_F['1sc_Fd'])-np.array(dict_F['1sc_Fd_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_F['1sc_Fs'],
                        label='S-state',
                        yerr=[lower_error_S, upper_error_S],
                        markeredgecolor='purple',
                        ecolor='purple',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
            ax.errorbar(x=x_axis,
                        y=dict_F['1sc_Fd'],
                        label='D-state',
                        yerr=[lower_error_D, upper_error_D],
                        markeredgecolor='green',
                        ecolor='green',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error_S = abs(np.array(dict_F['1sc_Fs'])-np.array(dict_F['1sc_Fs_stoch_bot']))
            upper_error_S = abs(np.array(dict_F['1sc_Fs'])-np.array(dict_F['1sc_Fs_stoch_top']))
            lower_error_D = abs(np.array(dict_F['1sc_Fd'])-np.array(dict_F['1sc_Fd_stoch_bot']))
            upper_error_D = abs(np.array(dict_F['1sc_Fd'])-np.array(dict_F['1sc_Fd_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_F['1sc_Fs'],
                        label='D-state',
                        yerr=[lower_error_S, upper_error_S],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
            ax.errorbar(x=x_axis,
                        y=dict_F['1sc_Fd'],
                        label='D-state',
                        yerr=[lower_error_D, upper_error_D],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_F['1sc_Fs'], color='purple', label='S-state')
        ax.plot(x_axis, dict_F['1sc_Fd'], color='green', label='D-state')
        if show_osc_error:
            ax.fill_between(x_axis, dict_F['1sc_Fs_osc_bot'], dict_F['1sc_Fs_osc_top'],
                            facecolor='purple', alpha=0.3)
            ax.fill_between(x_axis,dict_F['1sc_Fd_osc_bot'], dict_F['1sc_Fd_osc_top'],
                            facecolor=d_shades['osc'], alpha=d_shades['alpha'])
        if show_stoch_error:
            ax.fill_between(x_axis, dict_F['1sc_Fs_stoch_bot'], dict_F['1sc_Fs_stoch_top'], 
                            facecolor='purple', alpha=0.5)
            ax.fill_between(x_axis, dict_F['1sc_Fd_stoch_bot'], dict_F['1sc_Fd_stoch_top'], 
                            facecolor=d_shades['stoch'], alpha=d_shades['alpha'])    

        
    # 2sc
    ax = ax_F[1]
    #ax.set_xlabel(f'{variable_hyperparam_name}', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    #ax.plot(x_axis, dict_F['2sc_Fs'], color='purple')
    #ax.plot(x_axis, dict_F['2sc_Fd'], color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    
    # Errors
    if shades_or_bars == 'bars':
        if show_osc_error:
            lower_error_S = abs(np.array(dict_F['2sc_Fs'])-np.array(dict_F['2sc_Fs_osc_bot']))
            upper_error_S = abs(np.array(dict_F['2sc_Fs'])-np.array(dict_F['2sc_Fs_osc_top']))
            lower_error_D = abs(np.array(dict_F['2sc_Fd'])-np.array(dict_F['2sc_Fd_osc_bot']))
            upper_error_D = abs(np.array(dict_F['2sc_Fd'])-np.array(dict_F['2sc_Fd_osc_top']))
            ax.errorbar(x=x_axis,
                        y=dict_F['2sc_Fs'],
                        label='S-state',
                        yerr=[lower_error_S, upper_error_S],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
            ax.errorbar(x=x_axis,
                        y=dict_F['2sc_Fd'],
                        label='S-state',
                        yerr=[lower_error_D, upper_error_D],
                        markeredgecolor='black',
                        ecolor='blue',
                        fmt='x',
                        capsize=5.,
                        markersize=5.)
        if show_stoch_error:
            lower_error_S = abs(np.array(dict_F['2sc_Fs'])-np.array(dict_F['2sc_Fs_stoch_bot']))
            upper_error_S = abs(np.array(dict_F['2sc_Fs'])-np.array(dict_F['2sc_Fs_stoch_top']))
            lower_error_D = abs(np.array(dict_F['2sc_Fd'])-np.array(dict_F['2sc_Fd_stoch_bot']))
            upper_error_D = abs(np.array(dict_F['2sc_Fd'])-np.array(dict_F['2sc_Fd_stoch_top']))
            ax.errorbar(x=x_axis,
                        y=dict_F['2sc_Fs'],
                        label='S-state',
                        yerr=[lower_error_S, upper_error_S],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
            ax.errorbar(x=x_axis,
                        y=dict_F['2sc_Fd'],
                        label='S-state',
                        yerr=[lower_error_D, upper_error_D],
                        color='orange',
                        fmt='x',
                        capsize=5.,
                        markersize=0.1)
    elif shades_or_bars == 'shades':
        ax.plot(x_axis, dict_F['2sc_Fs'], color='purple')
        ax.plot(x_axis, dict_F['2sc_Fd'], color='green')
        if show_osc_error:
            ax.fill_between(x_axis, dict_F['2sc_Fs_osc_bot'], dict_F['2sc_Fs_osc_top'],
                            facecolor='purple', alpha=0.3)
            ax.fill_between(x_axis,dict_F['2sc_Fd_osc_bot'], dict_F['2sc_Fd_osc_top'],
                            facecolor=d_shades['osc'], alpha=d_shades['alpha'])
        if show_stoch_error:
            ax.fill_between(x_axis, dict_F['2sc_Fs_stoch_bot'], dict_F['2sc_Fs_stoch_top'], 
                            facecolor='purple', alpha=0.5)
            ax.fill_between(x_axis, dict_F['2sc_Fd_stoch_bot'], dict_F['2sc_Fd_stoch_top'], 
                            facecolor=d_shades['stoch'], alpha=d_shades['alpha'])    
    
       
    # 1sd
    ax = ax_F[2]
    #ax.set_xlabel(f'{variable_hyperparam_name}', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    ax.plot(x_axis, dict_F['1sd_Fs'], color='purple')
    ax.plot(x_axis, dict_F['1sd_Fd'], color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis, dict_F['1sd_Fs_osc_bot'], dict_F['1sd_Fs_osc_top'], 
                        facecolor='purple', alpha=0.3)
        ax.fill_between(x_axis, dict_F['1sd_Fd_osc_bot'], dict_F['1sd_Fd_osc_top'], 
                        facecolor=d_shades['osc'], alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis, dict_F['1sd_Fs_stoch_bot'], dict_F['1sd_Fs_stoch_top'], 
                        edgecolor='purple', facecolor='purple', alpha=0.5)
        ax.fill_between(x_axis, dict_F['1sd_Fd_stoch_bot'], dict_F['1sd_Fd_stoch_top'], 
                        facecolor=d_shades['stoch'], alpha=d_shades['alpha'])
        
    # 2sd
    ax = ax_F[3]
    #ax.set_xlabel(f'{variable_hyperparam_name}', fontsize=15)
    ax.tick_params(axis='both', labelsize=14)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.tick_params(axis='both', which='major', width=1.7, length=4.5)
    
    ax.plot(x_axis, dict_F['2sd_Fs'], color='purple')
    ax.plot(x_axis, dict_F['2sd_Fd'], color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis, dict_F['2sd_Fs_osc_bot'], dict_F['2sd_Fs_osc_top'], 
                        facecolor='purple', alpha=0.3)
        ax.fill_between(x_axis, dict_F['2sd_Fd_osc_bot'], dict_F['2sd_Fd_osc_top'], 
                        facecolor=d_shades['osc'], alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis, dict_F['2sd_Fs_stoch_bot'], dict_F['2sd_Fs_stoch_top'], 
                        facecolor='purple', alpha=0.5)
        ax.fill_between(x_axis, dict_F['2sd_Fd_stoch_bot'], dict_F['2sd_Fd_stoch_top'], 
                        facecolor=d_shades['stoch'], alpha=d_shades['alpha'])
    
    
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=5,  
               fancybox=True, shadow=True, fontsize=14)
    
    
    if save_plot:
        plt.savefig(path_of_plot, format='pdf', bbox_inches='tight')
        print(f'Figure saved in {path_of_plot}')
    
    plt.show()