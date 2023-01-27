# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This script computes the convergence rate and energy of trained models. The 
rate is computed as the number of models that have converged within the range 
E <= E_min MeV divided by the total number of models that have been trained. If
necessary, the data is stored under the 'error_data/' folder.

· In Section 'ADJUSTABLE PARAMETERS' we can set the values of the variables 
according to our preferences. Below is an explanation of what each variable 
does.

    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    save_data --> Boolean, Sets whether the data is saved or not.
    E_min --> float, (mean) Energy below which we stop accepting runs.
    print_info --> Boolean, Whether to print out to the console the 
                    convergence rates and energy of every model or not.
    
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
    
· In Section 'FILE SEARCHING' we specify a hyperparameter combinatioon and we 
search all trained models that match that criterium. We then count the number
of trained models and also the number of converged models, and we compute the
convergence rate. The data is then stored under the 'error_data/' folder. 
"""

############# IMPORTS #############
import os, sys, pathlib, statistics, math
from itertools import product

initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

############# ADJUSTABLE PARAMETERS #############
# General parameters
net_archs = ['1sc', '2sc', '1sd', '2sd']
save_data = True
E_min = -2.22
n_models = 150
print_info = True

# Training hyperparameters
hidden_neurons = [20, 30, 40, 60, 80, 100]
actfuns = ['Sigmoid', 'Softplus', 'ReLU']
optimizers = ['RMSprop']
learning_rates = [0.005, 0.01, 0.05] # Use decimal notation 
smoothing_constant = [0.7, 0.8, 0.9]
momentum = [0.0, 0.9]

# Default hyperparameters
default_actfun = 'Sigmoid'
default_optim = 'RMSprop'
default_lr = 0.01
default_alpha = 0.9
default_momentum = 0.0

############# FILE SEARCHING #############
for arch, nhid, optim, actfun, lr, alpha, mom in product(net_archs, 
                                                         hidden_neurons, 
                                                         optimizers, actfuns, 
                                                         learning_rates, 
                                                         smoothing_constant, 
                                                         momentum): 
    nhid = 32 if arch == '2sd' and nhid == 30 else nhid
            # Default hyperparams
    if not ((optim == default_optim and
             actfun == default_actfun and
             lr == default_lr and
             alpha == default_alpha and
             mom == default_momentum) or
            (nhid == 60 and
            # Variable lr
            ((optim == default_optim and
            actfun == default_actfun and
            alpha == default_alpha and
            mom == default_momentum) or
            # Variable actfun
            (optim == default_optim and
            lr == default_lr and
            alpha == default_alpha and
            mom == default_momentum) or
            # Variable alpha
            (optim == default_optim and
            actfun == default_actfun and
            lr == default_lr and
            mom == default_momentum) or    
            # Variable momentum
            (optim == default_optim and
            actfun == default_actfun and
            lr == default_lr and
            alpha == default_alpha)))):
            continue
                                                                             
    try:        
        ###### ENERGIES ######
        path_to_filtered_files = f'../error_data/{arch}/nhid{nhid}/' \
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/filtered_runs/'
        energies = []
        E_top_errors = []
        E_bot_errors = []
        E_errors = [E_bot_errors, E_top_errors]
        for file in os.listdir(path_to_filtered_files):
            if (os.path.isfile(f'{path_to_filtered_files}{file}') and 
            file.split('.')[1] == 'txt' and 
            file == 'means_and_errors.txt'):
                with open(f'{path_to_filtered_files}{file}', 'r') as f:
                    for line in f.readlines():        
                        energies.append(float(line.split(' ')[0]))
                        E_top_errors.append(float(line.split(' ')[1]))
                        E_bot_errors.append(float(line.split(' ')[2]))
                    f.close()
        if len(energies) > 1:
            mean_E = sum(energies) / len(energies)
            stdev_E = statistics.stdev(energies) / math.sqrt(len(energies))
            mean_E_top_errors = sum(E_top_errors) / len(E_top_errors)
            mean_E_bot_errors = sum(E_bot_errors) / len(E_bot_errors)
        else:
            print(f'energies has {len(energies)} elements, so we cannot' \
                  ' compute the statistics...')
    
        ###### CONVERGED MODELS ######
        n_converged = 0
        path_to_error_files = f'../error_data/{arch}/nhid{nhid}/' \
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/'
        for file in os.listdir(path_to_error_files):
            if (os.path.isfile(f'{path_to_error_files}{file}') and 
            file.split('.')[1] == 'txt' and 
            file == 'means_and_errors.txt'):
                with open(f'{path_to_error_files}{file}', 'r') as f:
                    for line in f.readlines():        
                        E = line.strip('\n').split(' ')[0]
                        if float(E) <= E_min:
                            n_converged += 1
                    f.close()
     
        ###### TRAINED MODELS ######
        path_to_trained_models = f'../../saved_models/n3lo/{arch}/nhid{nhid}/'\
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/'
        n_trained = len(os.listdir(path_to_trained_models))
        if n_trained < n_models:
            print(f'Models at {path_to_trained_models} have only ' \
                  f'{n_trained}/150 trained models.')
        conv_rate = n_converged/n_trained
        
        if conv_rate > 1.:
            raise Warning('Oops! It seems that there are more converged ' \
                            f'models at {path_to_error_files} than trained ' \
                            f'models at {path_to_trained_models}')
                
        if print_info:
            print(f'\nArch = {arch}, Neurons = {nhid}, Actfun = {actfun}, ' \
                    f'lr = {lr}, Alpha = {alpha}, Mu = {mom}, ' \
                    f'<E> = {mean_E}, E+ = ({stdev_E}, {mean_E_top_errors}), '\
                    f'E- = ({stdev_E}, {mean_E_bot_errors}), ' \
                    'Rate = {:.2f}%'.format(100*conv_rate))
        
        ###### DATA SAVING ######
        # We save the convergence rate data to a file
        if save_data:
            conv_error_file = path_to_error_files + 'conv_rate.txt'
            # If the file we want to create already exists, we first delete it
            with open(conv_error_file, 'w') as file: 
                # n_converged n_trained conv_rate E E_top_stoch E_top_osc 
                # E_bot_stoch E_bot_osc
                file.write(f'{n_converged} {n_trained} {conv_rate} {mean_E} ' \
                           f'{stdev_E} {mean_E_top_errors} {stdev_E} ' \
                           f'{mean_E_bot_errors}')
                file.close()
    except FileNotFoundError:
        print('\nIt seems that there are no files under ' \
              f'{path_to_error_files}. Skipping this file...')
