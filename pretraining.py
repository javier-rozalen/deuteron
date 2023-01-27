# -*- coding: utf-8 -*-
#################### CONTENTS OF THE FILE ####################
"""
In this file we (pre)train an Artificial Neural Network to fit a given function
that resembles a wave function. We do this for different hyperparameters and 
network architectures. The trained models, as well as figures of the training
process, are stored under the folder 'saved_models/pretraining/'.

· In Section 'ADJUSTABLE PARAMETERS' we can change the values of the variables.
An extensive explanation is given below. 

    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    device --> 'cpu', 'cuda', Physical processor into which we load our model.
    nchunks_general --> int, Number of batches of pretrained models. This is 
                        only for parallelized computing. If using a single 
                        instance of this program, set nchunks_general to 1. 
    which_chunk --> int, Selects the desired batch (by number). If using 
                    multiple instances of this program, set this parameter to 
                    'int(sys.argv[1])', for example.
    save_model --> Boolean, Saves the model at the end of the training or not.
    save_plot --> Boolean, Saves the plot at the end of the training or not.
    epochs_general --> int, Number of training epochs. 
    periodic_plots --> Boolean, Sets whether plots of the wave function are 
                        periodically shown. 
    show_arch --> Boolean, Sets whether the network architecture is printed at 
                the beginning of each file.
    leap --> int, Number of epochs between updates and/or plots.
    hidden_neurons --> List, Contains different numbers of hidden neurons.
    seed_from --> int, Initial seed.
    seed_to --> int, Final seed.
    recompute --> Boolean, Whether to train a model which had been already
                trained.
    
    MESH PARAMETERS 
    -----------------------------------
    q_max --> float, Maximum value of the momentum axis
    n_samples --> int, Number of mesh points. DO NOT CHANGE THE DEFAULT VALUE.
    train_a --> float, Lower limit of momentum
    train_b --> float, Upper limit of momentum. It is set to q_max by default.
    
    TRAINING HYPERPARAMETERS
    -----------------------------------
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
    
· In Section 'DIRECTORY SUPPORT' we recursively create, if necessary, the 
directories and subdirectories where the trained models will be stored..

· In Section 'MESH SET PREPARATION' we create the lattice on which we train our
ANNs.

· In Section 'TARGET FUNCTION' we generate the function that we want to fit.

· In Section 'LOOP OVER SEEDS AND HYPERPARAMS' we specify the combination of
hyperparameters that we want to iterate over.

· In Section 'EPOCH LOOP' we carry out the actual ANN training, training models 
with specific hyperparameters, and we also store the data. 
"""

##################### IMPORTS #####################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

import torch, time, math
import numpy as np
from tqdm import tqdm
from itertools import product

# My modules
import modules.integration as integration
import modules.neural_networks as neural_networks
from modules.plotters import pretraining_plots
from modules.aux_functions import dir_support, show_layers, split
from modules.aux_functions import pretrain_loop
from modules.loss_functions import overlap

##################### ADJUSTABLE PARAMETERS #####################
# General parameters
net_archs = ['1sc', '2sc', '1sd', '2sd']
device = 'cpu' 
nchunks_general = 1
which_chunk = int(sys.argv[1]) if nchunks_general != 1 else 0
save_model = True
save_plot = True
epochs_general = 2000
periodic_plots = False
show_arch = False
leap = 500
# The code handles the 30 <--> 32 conversion in hidden neurons
hidden_neurons = [20, 30, 40, 60, 80, 100] 
seed_from = 0
seed_to = 150
recompute = False

# Mesh parameters
q_max = 500
n_samples = 64 # Do not change this.
train_a = 0
train_b = q_max

# Training hyperparameters
actfuns = ['Sigmoid', 'Softplus', 'ReLU']
optimizers = ['RMSprop']
learning_rates = [0.005, 0.01, 0.05] # Use decimal notation 
epsilon = 1e-8
smoothing_constant = [0.7, 0.8, 0.9]
momentum = [0.0, 0.9]

# Default hyperparam values
default_actfun = 'Sigmoid'
default_optim = 'RMSprop'
default_lr = 0.01
default_alpha = 0.9
default_momentum = 0.0

################### DIRECTORY SUPPORT ###################
# Directory structure
path_steps_models = ['saved_models',
                     'pretraining',
                     'arch',
                     'nhid',
                     'optimizer',
                     'actfun',
                     'lr',
                     'smoothing_constant',
                     'momentum',
                     'models/plots']

nsteps = range(len(path_steps_models)) 

################### MESH SET PREPARATION ###################
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

# Integration-specific
x_i_int = [torch.tensor(float(e)) for e in x]
w_i_int = [torch.tensor(float(e)) for e in w]

x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # q mesh from 0 to 1
# tangential q mesh from 0 to q_max
k = [(q_max/math.tan(x_i[-1]*math.pi/2))* \
     math.tan(float(e)*math.pi/2) for e in x_i] 
w_i = [torch.tensor(float(e)/2) for e in w] # GL weights
cos2 = [1/(math.cos(float(x_i[i])*math.pi/2))**2 for i in range(n_samples)]
p = (q_max/math.tan(x_i[-1]*math.pi/2))*math.pi/2 
w_i = [p*w_i[r]*cos2[r] for r in range(n_samples)]
w_i = torch.stack(w_i)
Q_train = torch.tensor(k) # Momentum mesh
q_2 = Q_train**2 # Squared momentum mesh
Q_test = Q_train

################### TARGET FUNCTION ###################
psi_ansatz_s = torch.exp((-1.5**2)*Q_train**2/2) # target function L=0
psi_ansatz_d = (Q_train**2)*torch.exp((-1.5**2)*Q_train**2/2) # "" "" L=2

norm_s = integration.gl64(w_i,q_2*(psi_ansatz_s**2))
norm_d = integration.gl64(w_i,q_2*(psi_ansatz_d**2))

psi_ansatz_s_normalized = psi_ansatz_s/torch.sqrt(norm_s)
psi_ansatz_d_normalized = psi_ansatz_d/torch.sqrt(norm_d)

################### LOOP OVER SEEDS AND HYPERPARAMS ###################
start_time_all = time.time()
for arch, Nhid, optim, actfun, lr, alpha, mom in product(net_archs, 
                                                         hidden_neurons,
                                                         optimizers, actfuns,
                                                         learning_rates,
                                                         smoothing_constant,
                                                         momentum):
    Nhid = 32 if arch == '2sd' and Nhid == 30 else Nhid
    epochs = epochs_general            
        
    # We restrict the pretraining to a specific hyperparameter configuration
            # Default hyperparams
    if not ((optim == default_optim and
             actfun == default_actfun and
             lr == default_lr and
             alpha == default_alpha and
             mom == default_momentum) or
            (Nhid == 60 and
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
            alpha == default_alpha and not
            (arch == '2sd' and mom == 0.9))))):
        # print('Skipping non-whitelisted hyperparam combination...')
        continue
    
    # Directory support
    for _ in nsteps:
        if save_model:
            path_steps_models = ['saved_models',
                                 'pretraining',
                                 f'{arch}',
                                 f'nhid{Nhid}',
                                 f'optim{optim}',
                                 f'{actfun}',
                                 f'lr{lr}', 
                                 f'alpha{alpha}', 
                                 f'mu{mom}',
                                 'models']
            dir_support(path_steps_models)
        if save_plot:
            path_steps_plots = ['saved_models',
                                'pretraining',
                                f'{arch}',
                                f'nhid{Nhid}',
                                f'optim{optim}',
                                f'{actfun}',
                                f'lr{lr}', 
                                f'alpha{alpha}', 
                                f'mu{mom}',
                                'plots']
            dir_support(path_steps_plots)   
            
    # We reduce the list of seeds to iterate over to those seeds that have not
    # been already used
    l_seeds = range(seed_from, seed_to + 1)
    l_pretrained_seeds = []
    if recompute == False:
        path_to_pretrained_models = '/'.join(path_steps_models) + '/'
        list_of_pretrained_models = os.listdir(path_to_pretrained_models)
        for pm in list_of_pretrained_models:
            seed = int(pm.split('_')[0].replace('seed', ''))
            if seed in l_seeds:
                l_seeds.remove(seed)
    l_seeds.sort()
    n_seeds = len(l_seeds)
    
    # We adjust the number of processes to the number of models
    if nchunks_general > n_seeds:
        nchunks = n_seeds
    else:
        nchunks = nchunks_general
    if which_chunk >= nchunks:
        continue

    # We iterate over the list of seeds we want to (pre)train
    for seed in split(l=l_seeds, n=nchunks)[which_chunk]:
        saved_models_dir = '/'.join(path_steps_models) + '/' 
        name_without_dirs = f'seed{seed}_epochs{epochs}.pt'
        # If the current model has been already pretrained, we skip it
        if (recompute == False and 
            name_without_dirs in os.listdir(saved_models_dir)): 
            continue
        
        print(f'\nArch = {arch}, Neurons = {Nhid}, Optimizer = {optim},' \
              f' Actfun = {actfun}, lr = {lr}, Alpha = {alpha}, Mu = {mom},' \
                  f'Seed = {seed}/{seed_to}')
        torch.manual_seed(seed)
        path_model = f'saved_models/pretraining/{arch}/nhid{Nhid}/' \
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/' \
            f'models/seed{seed}_epochs{epochs}'
        path_plot = f'saved_models/pretraining/{arch}/nhid{Nhid}/' \
            f'optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/' \
            f'plots/seed{seed}_epochs{epochs}'
    
        # ANN Parameters
        if arch == '1sc':
            Nhid_prime = Nhid
        elif arch == '2sc' or arch == '1sd':
            Nhid_prime = int(Nhid/2)
        elif arch == '2sd':
            Nhid_prime = int(Nhid/4)
        Nin = 1
        Nout = 1 if arch == '2sd' else 2 
        W1 = torch.rand(Nhid_prime, Nin, requires_grad=True)*(-1.) 
        B = torch.rand(Nhid_prime)*2. - torch.tensor(1.) 
        W2 = torch.rand(Nout, Nhid_prime, requires_grad=True) 
        Ws2 = torch.rand(1, Nhid_prime, requires_grad=True) 
        Wd2 = torch.rand(1, Nhid_prime, requires_grad=True) 
        
        # We load our psi_ann to the CPU (or GPU)
        net_arch_map = {'1sc':neural_networks.sc_1,
                        '2sc':neural_networks.sc_2,
                        '1sd':neural_networks.sd_1,
                        '2sd':neural_networks.sd_2}
        psi_ann = net_arch_map[arch](Nin=Nin,
                                     Nhid_prime=Nhid_prime,
                                     Nout=Nout,
                                     W1=W1,
                                     Ws2=Ws2,
                                     B=B,
                                     W2=W2,
                                     Wd2=Wd2,
                                     actfun=actfun).to(device)
        
        # We define the loss function and the optimizer
        loss_fn = overlap
        optimizer = getattr(torch.optim, optim)(params=psi_ann.parameters(),
                                        lr=lr,
                                        eps=epsilon,
                                        alpha=alpha,
                                        momentum=mom)
        if show_arch:
            show_layers(psi_ann)
                                      
        ##################### EPOCH LOOP #####################        
        # We store the energy data in lists for later plotting
        overlap_s, overlap_d = [], []
        
        start_time = time.time()
        for t in tqdm(range(epochs)):  
            (psi_s_pred, psi_d_pred, 
            k_s, k_d) = pretrain_loop(model=psi_ann,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      arch=arch,
                                      train_data=Q_train,
                                      q_2=q_2,
                                      integration=integration,
                                      w_i=w_i,
                                      norm_s=norm_s,
                                      norm_d=norm_d,
                                      psi_ansatz_s=psi_ansatz_s,
                                      psi_ansatz_d=psi_ansatz_d) 
            overlap_s.append(k_s.item())
            overlap_d.append(k_d.item())
        
            if ((t+1) % leap) == 0 and periodic_plots:
                pretraining_plots(x_axis=Q_test,
                                  psi_s_pred=psi_s_pred,
                                  psi_d_pred=psi_d_pred,
                                  n_samples=n_samples,
                                  psi_ansatz_s_normalized=
                                  psi_ansatz_s_normalized,
                                  psi_ansatz_d_normalized=
                                  psi_ansatz_d_normalized,
                                  overlap_s=overlap_s,
                                  overlap_d=overlap_d,
                                  path_plot=path_plot,
                                  t=t,
                                  s=save_plot if t == epochs -1 else False,
                                  show=False) 
                
        print('Model pretrained!')
        print('Total execution time: {:6.2f} seconds (run on {})'.format(
            time.time()-start_time_all, device))
        
        full_path_model = '{}.pt'.format(path_model)
        full_path_plot = '{}.pdf'.format(path_plot)
        
        if save_model:
            state_dict = {'model_state_dict':psi_ann.state_dict(),
                          'optimizer_state_dict':optimizer.state_dict()}
            torch.save(state_dict, full_path_model)
            print(f'Model saved in {full_path_model}')
        if save_plot: 
            print(f'Plot saved in {full_path_plot}')                   
    
print("\nAll done! :)")