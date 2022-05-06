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
    show_stoch_error --> Boolena, (self-explanatory)
    learning_rate --> float, Learning rate.
"""

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
save_plot = True
path_of_plot = 'saved_plots/energy_plot.pdf'
show_osc_error = True
show_stoch_error = True
learning_rate = 0.01 # Use decimal notation.

###################################### IMPORTS ######################################
import matplotlib.pyplot as plt
import math, os, re, statistics
from math import log10, floor
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import (MultipleLocator)

###################################### MISC. ######################################
net_arch_name_map = {'1sc':'fully_connected_ann','2sc':'fully_connected_ann',
                     '1sd':'separated_ann','2sd':'separated_ann'}

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

if not os.path.exists('plotting_data'):
    os.makedirs('plotting_data')
if not os.path.exists('saved_plots'):
    os.makedirs('saved_plots')

############################## FURTHER ERROR COMPUTATIONS #################################
for network_arch in list(net_arch_name_map.keys()):
    # We look for the (filtered) error files.
    path_to_error_files = f'../error_analysis/error_data/nlayers{network_arch[0]}/{net_arch_name_map[network_arch]}/Sigmoid/lr{learning_rate}/filtered_runs/'
    
    # joint_graph_file creation    
    joint_graph_file = '{}joint_graph.txt'.format(path_to_error_files)
    
    # We check wether 'joint_graph_file.txt' exists and we delete it if it does
    if os.path.isfile(joint_graph_file):
        os.remove(joint_graph_file)
        
    with open(f'plotting_data/master_plot_data_{network_arch}.txt','w') as global_file:
        global_file.close()
            
    all_files = os.listdir(path_to_error_files)
    list_of_error_files = []
    # We select the error files among all the files in this directory
    for file in all_files:
        if len(file.split('.'))>1:
            if file.split('.')[1] == 'txt' and file[:4] == 'nhid':
                list_of_error_files.append(file)
    
    # We sort the list of pretrained models by nhid
    def num_sort(test_string):
        return list(map(int, re.findall(r'\d+', test_string)))[0]
    list_of_error_files.sort(key=num_sort)

    hidden_neurons = []
    for file in list_of_error_files:
    
        ################## ERROR DATA FETCHING ##################
        filename = '{}{}'.format(path_to_error_files,file) 
        
        Nhid = int(file.split('.')[0].replace('nhid',''))
        hidden_neurons.append(Nhid)
        test_number = []
    
        energies = []
        E_top_errors = []
        E_bot_errors = []
        E_errors = [E_bot_errors,E_top_errors]
        
        ks = []
        ks_top_errors = []
        ks_bot_errors = []
        
        kd = []
        kd_top_errors = []
        kd_bot_errors = []
        
        pd = []
        pd_top_errors = []
        pd_bot_errors = []
        
        # We fill the lists above with the parameters computed with error_measure.py and later filtered with filter.py
        with open(filename, 'r') as file:
            c = 1
            for line in file.readlines():
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
                test_number.append(c)
                c += 1
            file.close()
        
        ################## FURTHER ERRORS COMPUTATION ##################  
        ks_errors = [ks_bot_errors,ks_top_errors]
        kd_errors = [kd_bot_errors,kd_top_errors]
        pd_errors = [pd_bot_errors,pd_top_errors]
        
        if len(energies)>0:
            
            mean_E = sum(energies)/len(energies)
            stdev_E = statistics.stdev(energies)/math.sqrt(len(energies))
            mean_E_top_errors = sum(E_top_errors)/len(E_top_errors)
            mean_E_bot_errors = sum(E_bot_errors)/len(E_bot_errors)
            
            mean_ks = sum(ks)/len(ks)
            stdev_ks = statistics.stdev(ks)/math.sqrt(len(ks))
            mean_ks_top_errors = sum(ks_top_errors)/len(ks_top_errors)
            mean_ks_bot_errors = sum(ks_bot_errors)/len(ks_bot_errors)
            
            mean_kd = sum(kd)/len(kd)
            stdev_kd = statistics.stdev(kd)/math.sqrt(len(kd)) # Stochastic error (used for both upper and lower bounds)
            mean_kd_top_errors = sum(kd_top_errors)/len(kd_top_errors) # Oscillating error (upper bound)
            mean_kd_bot_errors = sum(kd_bot_errors)/len(kd_bot_errors) # Oscillating error (lower bound)
                    
            # This saves all the parameters necessary to compute the final graph to the file 'joint_graph.txt'
            with open(joint_graph_file, 'a') as file:
                # E, E+, E-, Ks, Ks+, Ks-, Pd, Pd+, Pd-
                file.write(str(mean_E)+' '+
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
                file.close()
    
    ################## FURTHER ERRORS DATA FETCHING ##################
    """We read the data of joint_graph_file, and compute the necessary stuff for the fancy joint plot."""
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
    with open(joint_graph_file, 'r') as file:
        for line in file.readlines():
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
        file.close()
        
    # We generate lists with the complete error data.
    E_errors_joint = [E_bot_errors_joint,E_top_errors_joint]    
    ks_errors_joint = [ks_bot_errors_joint,ks_top_errors_joint]
    kd_errors_joint = [kd_bot_errors_joint,kd_top_errors_joint]
    
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
        
    # We save the data in a global file.
    with open(f'plotting_data/master_plot_data_{network_arch}.txt','a') as global_file:
        for E,E_osc_bot,E_osc_top,E_stoch_bot,E_stoch_top,Fs,Fs_osc_bot,Fs_osc_top,Fs_stoch_bot,Fs_stoch_top,Fd,Fd_osc_bot,Fd_osc_top,Fd_stoch_bot,Fd_stoch_top in zip(energies_joint,E_osc_shade_bot,E_osc_shade_top,
                                                                                                          E_stoch_shade_bot,E_stoch_shade_top,
                                                                                                          ks_joint,ks_osc_shade_bot,ks_osc_shade_top,
                                                                                                          ks_stoch_shade_bot,ks_stoch_shade_top,
                                                                                                          kd_joint,kd_osc_shade_bot,kd_osc_shade_top,
                                                                                                          kd_stoch_shade_bot,kd_stoch_shade_top):
            global_file.write(f'{E} {E_osc_bot} {E_osc_top} {E_stoch_bot} {E_stoch_top} {Fs} {Fs_osc_bot} {Fs_osc_top} {Fs_stoch_bot} {Fs_stoch_top} {Fd} {Fd_osc_bot} {Fd_osc_top} {Fd_stoch_bot} {Fd_stoch_top}\n')
        global_file.close() 
            
    print(f'Error data of model {network_arch} successfully appended to master_plot_data_{network_arch}.txt.')
        
################################# DATA PREPARATION FOR PLOTTING #################################
if True:
    print('\n')
    data_list = ['E','E_osc_bot','E_osc_top','E_stoch_bot','E_stoch_top',
                 'Fs','Fs_osc_bot','Fs_osc_top','Fs_stoch_bot','Fs_stoch_top',
                 'Fd','Fd_osc_bot','Fd_osc_top','Fd_stoch_bot','Fd_stoch_top']
    
    dict_1sc,dict_2sc,dict_1sd,dict_2sd,dict_E,dict_F = dict(),dict(),dict(),dict(),dict(),dict()
    dict_dict = {'1sc':dict_1sc,'2sc':dict_2sc,'1sd':dict_1sd,'2sd':dict_2sd}
    
    # We create the keys
    for e in data_list:
        key_1sc = f'1sc_{e}'
        key_2sc = f'2sc_{e}'
        key_1sd = f'1sd_{e}'
        key_2sd = f'2sd_{e}'
        dict_1sc[key_1sc],dict_2sc[key_2sc],dict_1sd[key_1sd],dict_2sd[key_2sd] = list(),list(),list(),list()
        
    for network_arch in list(net_arch_name_map.keys()):
        print(f'Loading dictionary of model {network_arch}...')
        with open(f'plotting_data/master_plot_data_{network_arch}.txt','r') as f:
            list_of_keys = list(dict_dict[network_arch].keys())
            for line in f.readlines():
                line = line.split(' ')
                c = 0
                for item in line:
                    key_at_index = list_of_keys[c]
                    dict_dict[network_arch][key_at_index].append(float(item))
                    c += 1
            f.close()         
            
        # We fill a dictionary with the energies and another one with the fidelities
        for e in list(dict_dict[network_arch].keys()):
            if e.split('_')[1] == 'E':
                dict_E[e] = dict_dict[network_arch][e]
            else:
                dict_F[e] = dict_dict[network_arch][e]
            
################################# PLOTTING #################################
if True:
    fig, axes = plt.subplots(nrows=2,ncols=4,figsize=(14,5),sharey='row',sharex='col')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.05,hspace=0.1)
    ax_E = axes[0]
    ax_F = axes[1]
    x_axis = [20, 30, 40, 60, 80, 100]
    
    ### Energies ###
    # Fc 1L
    ax = ax_E[0]
    ax.set_title('1 sc',fontsize=17)
    ax.set_ylabel('E (Mev)',fontsize=15)
    ax.set_ylim(-2.227,-2.220)
    #ax.yaxis.set_ticklabels(['-2.227','-2.226','-2.225','-2.224','-2.223','-2.222','-2.221','-2.220'],fontsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    
    ax.plot(x_axis,dict_E['1sc_E'],label='Energy')
    ax.axhline(y=-2.2267, color="red", linestyle="--",label='Exact')
    if show_osc_error:
        ax.fill_between(x_axis,dict_E['1sc_E_osc_bot'],dict_E['1sc_E_osc_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.3)
    if show_stoch_error:
        ax.fill_between(x_axis,dict_E['1sc_E_stoch_bot'],dict_E['1sc_E_stoch_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.5) 
    
    #plt.setp(ax.get_yticklabels(), visible=True)
    
    # Fc 2L
    ax = ax_E[1]
    ax.set_title('2 sc',fontsize=17)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_E['2sc_E'])
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_E['2sc_E_osc_bot'],dict_E['2sc_E_osc_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.3)
    if show_stoch_error:
        ax.fill_between(x_axis,dict_E['2sc_E_stoch_bot'],dict_E['2sc_E_stoch_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.5)
    
    # Sep 1L
    ax = ax_E[2]
    ax.set_title('1 sd',fontsize=17)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_E['1sd_E'])
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_E['1sd_E_osc_bot'],dict_E['1sd_E_osc_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.3)
    if show_stoch_error:
        ax.fill_between(x_axis,dict_E['1sd_E_stoch_bot'],dict_E['1sd_E_stoch_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.5)
        
    # Sep 2L
    ax = ax_E[3]
    ax.set_title('2 sd',fontsize=17)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_E['2sd_E'])
    ax.axhline(y=-2.2267, color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_E['2sd_E_osc_bot'],dict_E['2sd_E_osc_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.3)
    if show_stoch_error:
        ax.fill_between(x_axis,dict_E['2sd_E_stoch_bot'],dict_E['2sd_E_stoch_top'],edgecolor='#089FFF',facecolor='#089FFF',alpha=0.5) 
    
    ### Fidelities ###
    d_shades = {'stoch':'#249225','osc':'#abd5ab','alpha':0.5}
    
    # Fc 1L
    ax = ax_F[0]
    ax.set_ylabel('Fidelity',fontsize=15)
    ax.set_ylim(0.99992,1.000005)
    #ax.yaxis.set_ticklabels(['0.99990','0.99992','0.99994','0.99996','0.99998','1.00000',],fontsize=14)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
    ax.set_xlabel('$N_{\mathrm{hid}}$',fontsize=15)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    ax.yaxis.set_minor_locator(MultipleLocator(0.00001))
    
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_F['1sc_Fs_osc_bot'],dict_F['1sc_Fs_osc_top'],facecolor='purple',alpha=0.3)
        ax.fill_between(x_axis,dict_F['1sc_Fd_osc_bot'],dict_F['1sc_Fd_osc_top'],facecolor=d_shades['osc'],alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis,dict_F['1sc_Fs_stoch_bot'],dict_F['1sc_Fs_stoch_top'],facecolor='purple',alpha=0.5)
        ax.fill_between(x_axis,dict_F['1sc_Fd_stoch_bot'],dict_F['1sc_Fd_stoch_top'],facecolor=d_shades['stoch'],alpha=d_shades['alpha'])
    ax.plot(x_axis,dict_F['1sc_Fs'],color='purple',label='S-state')
    ax.plot(x_axis,dict_F['1sc_Fd'],color='green',label='D-state')
        
    # Fc 2L
    ax = ax_F[1]
    ax.set_xlabel('$N_{\mathrm{hid}}$',fontsize=15)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_F['2sc_Fs'],color='purple')
    ax.plot(x_axis,dict_F['2sc_Fd'],color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_F['2sc_Fs_osc_bot'],dict_F['2sc_Fs_osc_top'],facecolor='purple',alpha=0.3)
        ax.fill_between(x_axis,dict_F['2sc_Fd_osc_bot'],dict_F['2sc_Fd_osc_top'],facecolor=d_shades['osc'],alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis,dict_F['2sc_Fs_stoch_bot'],dict_F['2sc_Fs_stoch_top'],facecolor='purple',alpha=0.5)
        ax.fill_between(x_axis,dict_F['2sc_Fd_stoch_bot'],dict_F['2sc_Fd_stoch_top'],facecolor=d_shades['stoch'],alpha=d_shades['alpha'])
    
       
    # Sep 1L
    ax = ax_F[2]
    ax.set_xlabel('$N_{\mathrm{hid}}$',fontsize=15)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_F['1sd_Fs'],color='purple')
    ax.plot(x_axis,dict_F['1sd_Fd'],color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_F['1sd_Fs_osc_bot'],dict_F['1sd_Fs_osc_top'],facecolor='purple',alpha=0.3)
        ax.fill_between(x_axis,dict_F['1sd_Fd_osc_bot'],dict_F['1sd_Fd_osc_top'],facecolor=d_shades['osc'],alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis,dict_F['1sd_Fs_stoch_bot'],dict_F['1sd_Fs_stoch_top'],edgecolor='purple',facecolor='purple',alpha=0.5)
        ax.fill_between(x_axis,dict_F['1sd_Fd_stoch_bot'],dict_F['1sd_Fd_stoch_top'],facecolor=d_shades['stoch'],alpha=d_shades['alpha'])
        
    # Sep 2L
    ax = ax_F[3]
    ax.set_xlabel('$N_{\mathrm{hid}}$',fontsize=15)
    ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='both',which='both',direction='in')
    ax.tick_params(axis='both',which='major',width=1.7,length=4.5)
    
    ax.plot(x_axis,dict_F['2sd_Fs'],color='purple')
    ax.plot(x_axis,dict_F['2sd_Fd'],color='green')
    ax.axhline(y=1., color="red", linestyle="--")
    if show_osc_error:
        ax.fill_between(x_axis,dict_F['2sd_Fs_osc_bot'],dict_F['2sd_Fs_osc_top'],facecolor='purple',alpha=0.3)
        ax.fill_between(x_axis,dict_F['2sd_Fd_osc_bot'],dict_F['2sd_Fd_osc_top'],facecolor=d_shades['osc'],alpha=d_shades['alpha'])
    if show_stoch_error:
        ax.fill_between(x_axis,dict_F['2sd_Fs_stoch_bot'],dict_F['2sd_Fs_stoch_top'],facecolor='purple',alpha=0.5)
        ax.fill_between(x_axis,dict_F['2sd_Fd_stoch_bot'],dict_F['2sd_Fd_stoch_top'],facecolor=d_shades['stoch'],alpha=d_shades['alpha'])
    
    
    fig.legend(loc='lower center',bbox_to_anchor=(0.5,-0.14), ncol=5, fancybox=True, shadow=True, fontsize=14)
    
    
    if save_plot:
        plt.savefig(path_of_plot,format='pdf',bbox_inches='tight')
        print(f'Figure saved in {path_of_plot}')
    
    plt.show()