# -*- coding: utf-8 -*-
#################### CONTENTS OF THE FILE ####################
"""
This program trains an Artificial Neural Network to find the deuteron wave 
function. The ANN  has one input and two outputs, and hidden layers with Nhid 
hidden neurons in total. A 64-points (preferentially) tangential mesh is used 
as the momentum space upon which we train the ANN. The potential used is N3LO.
The models are saved under the 'saved_models/n3lo/' folder.

· In Section 'DIRECTORY SUPPORT' we define the directory structure where all 
trained models will be stored.

· In Section 'MESH SET PREPARATION' we create the lattice on which we train our
ANNs.

· In Section 'BENCHMARK DATA FETCHING/COMPUTATION' we fetch and generate the 
necessary data against which we compare our models.

· In Section 'MISC' we create a dictionary with the different network 
architectures and define the loss function.

· In Section 'LOOP OVER PRETRAINED MODELS' we iterate over every pretrained
model and we decide whether to perform the full training or not. Besides, we 
recursively create, if necessary, the directories and subdirectories where 
the trained models will be stored.

· In Section 'EPOCH LOOP' we perform the actual training of the current loaded
pretrained model. At the end of this section we save the fully-trained model.
"""

#################### IMPORTS ####################
import os
import sys
import torch, time, math
import numpy as np
from tqdm import tqdm
from itertools import product
import argparse

current = os.path.dirname(os.path.realpath(__file__))
os.chdir(current)

# My modules
from modules.physical_constants import hbar, mu
import modules.N3LO as N3LO
import modules.integration as integration
import modules.neural_networks as neural_networks
from modules.plotters import minimisation_plots
from modules.dir_support import dir_support
from modules.aux_functions import train_loop, show_layers, split, strtobool
from modules.loss_functions import energy

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="deuteron.py",
    usage="python3 %(prog)s [options]",
    description="Trains two neural networks to match the deuteron S and D channels.",
    epilog="Author: J Rozalén Sarmiento",
)
parser.add_argument(
    "--dev",
    help="Hardware device on which the code will run (default: cpu)",
    default="cpu",
    choices=["cpu", "gpu"],
    type=str,
)
parser.add_argument(
    "--save-model",
    help="Whether to save the model after training or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--save-plot",
    help="Whether to save the plot after training or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "-e",
    help="Number of epochs for the pretraining (default: 2000)",
    default=250000,
    type=int,
)
parser.add_argument(
    "--periodic-plots",
    help="Whether to periodically plot the wave functions during training or "
    "not (default: False)",
    default=False,
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
    "--show-arch",
    help="Whether to display the NN architecture or not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--leap",
    help="Number of epochs between updates/plots (default: 500)",
    default=500,
    type=int,
)
parser.add_argument(
    "--recompute",
    help="Whether to recompute models which have been already trained and "
    "saved to disk (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--shards",
    help="Number of shards in which to split all computations (default: 1)",
    default=1,
    type=int,
)
parser.add_argument(
    "--shard-number",
    help="Shard number corresponding to this instance of the code (default: 0)",
    default=0,
    type=int,
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
parser.add_argument(
    "--compile",
    help="Whether to pre-compile the NN calculatiosn or not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--plot",
    help="Whether to show a plot of different metrics at the end of each "
    "training (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
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
args = parser.parse_args()

#################### ADJUSTABLE PARAMETERS ####################
# General parameters
net_archs = args.archs
device = args.dev
if device == "gpu":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("There was an error trying to use the GPU. Falling back to CPU...")
        device = "cpu"
shards = args.shards
shard_number = args.shard_number if args.shards != 1 else 0
save_model = args.save_model
save_plot = args.save_plot
epochs_general = args.e
periodic_plots = args.periodic_plots
show_arch = args.show_arch
leap = args.leap
recompute = args.recompute
show_last_plot = args.plot
hidden_neurons = args.hidden_nodes


# Mesh parameters
q_max = 500
n_samples = 64 # Do not change this.
train_a = 0
train_b = q_max

# Training hyperparameters
actfuns = args.activations
optimizers = args.optimizers
learning_rates = args.lrs  # Use decimal notation
epsilon = 1e-8
smoothing_constant = args.alphas
momentum = args.mus

# Default hyperparam values
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

#################### DIRECTORY SUPPORT ####################

path_steps = ['saved_models',
             'n3lo',
             'arch',
             'nhid',
             'optimizer',
             'actfun',
             'lr',
             'smoothing_constant',
             'momentum',
             'models/plots']
nsteps = range(len(path_steps))

################### MESH SET PREPARATION ###################
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

x_i = [torch.tensor(float(e)*0.5 + 0.5) for e in x] # q mesh from 0 to 1
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

################### BENCHMARK DATA FETCHING/COMPUTATION ###################
# N3LO potential
n3lo = N3LO.N3LO('deuteron_data/2d_vNN_J1.dat', 'deuteron_data/wfk.dat')
V_ij = n3lo.getPotential()

# Target wavefunction (A Rios, J W T Keeble)
psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)   
psi_target_s_norm = integration.gl64(q_2*(psi_target_s)**2, w_i)
psi_target_d_norm = integration.gl64(q_2*(psi_target_d)**2, w_i)
psi_target_s_normalized = psi_target_s/torch.sqrt(psi_target_s_norm + \
                                                  psi_target_d_norm)
psi_target_d_normalized = psi_target_d/torch.sqrt(psi_target_d_norm + \
                                                  psi_target_s_norm)

# Vectors for the cost function
q2_128 = torch.cat((q_2, q_2))
wi_128 = torch.cat((w_i, w_i))

# Exact diagonalization wavefunction
psi_exact_s = []
psi_exact_d = []
Q_train1 = []
with open('deuteron_data/wfk_exact_diagonalization.txt', 'r') as file:
    c = 0
    for line in file.readlines():
        if c >= 2:
            psi_exact_s.append(torch.tensor(float(line.split(' ')[4])))
            psi_exact_d.append(torch.tensor(float(line.split(' ')[6])))
            Q_train1.append(torch.tensor(float(line.split(' ')[2])))
        c += 1

psi_exact_s = torch.stack(psi_exact_s)
psi_exact_d = torch.stack(psi_exact_d)
psi_exact_s_norm = integration.gl64(w_i, q_2*(psi_exact_s)**2)
psi_exact_d_norm = integration.gl64(w_i, q_2*(psi_exact_d)**2)
psi_exact_s_normalized = psi_exact_s/torch.sqrt(psi_exact_s_norm + \
                                                psi_exact_d_norm)
psi_exact_d_normalized = psi_exact_d/torch.sqrt(psi_exact_s_norm + \
                                                psi_exact_d_norm)

# Energy obtained with the exact diagonalization wf
def get_energy_pd():
    """Returns the energy and probability of D-state of the 
    exact diagonalization wf"""
    
    psi_128 = torch.cat((psi_exact_s_normalized, psi_exact_d_normalized))
    y = psi_128*q2_128*wi_128
    U_exact = torch.matmul(y, torch.matmul(V_ij, y)) 
    K_exact = ((hbar**2)/mu)*torch.dot(wi_128, (psi_128*q2_128)**2)
    E_exact = K_exact + U_exact
    PD_exact = 100.*psi_exact_d_norm/(psi_exact_s_norm + psi_exact_d_norm)
    return E_exact, PD_exact

E_exact, PD_exact = get_energy_pd()[0], get_energy_pd()[1]

################### MISC. ###################
# Neural Networks reference dictionary
net_arch_map = {'1sc': neural_networks.sc_1,
                '2sc': neural_networks.sc_2,
                '1sd': neural_networks.sd_1,
                '2sd': neural_networks.sd_2}
loss_fn = energy
num_sort = lambda x : int(x.split('_')[0].replace('seed', ''))

################### LOOP OVER PRETRAINED MODELS ###################
start_time_all = time.time()
for arch, Nhid, optim, actfun, lr, alpha, mom in product(net_archs,
                                                         hidden_neurons,
                                                         optimizers, actfuns,
                                                         learning_rates,
                                                         smoothing_constant,
                                                         momentum):
    Nhid = 32 if arch == '2sd' and Nhid == 30 else Nhid
    epochs = 500000 if lr == 0.005 else epochs_general
    
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
            alpha == default_alpha)))):
        # print('Skipping non-whitelisted hyperparam combination...')
        continue

    # We check that there are actually pretrained models with the above
    # hyperparameters, and move on to the next configuration otherwise
    path_to_pretrained_model = '/'.join(['saved_models',
                                         'pretraining',
                                         f'{arch}',
                                         f'nhid{Nhid}',
                                         f'optim{optim}',
                                         f'{actfun}',
                                         f'lr{lr}',
                                         f'alpha{alpha}',
                                         f'mu{mom}',
                                         'models']) + '/'
    try:
        list_of_pretrained_models = os.listdir(path_to_pretrained_model)
    except FileNotFoundError:
        print(f'Missing pretrained models at {path_to_pretrained_model}')
        continue

    # Directory support
    for _ in nsteps:
        path_steps_models = ['saved_models',
                             'n3lo',
                             f'{arch}',
                             f'nhid{Nhid}',
                             f'optim{optim}',
                             f'{actfun}',
                             f'lr{lr}', 
                             f'alpha{alpha}', 
                             f'mu{mom}',
                             'models']
        path_steps_plots = ['saved_models',
                            'n3lo',
                            f'{arch}',
                            f'nhid{Nhid}',
                            f'optim{optim}',
                            f'{actfun}',
                            f'lr{lr}', 
                            f'alpha{alpha}', 
                            f'mu{mom}',
                            'plots']
        dir_support(path_steps_models)
        dir_support(path_steps_plots)   

    saved_models_dir = '/'.join(path_steps_models) + '/'
    list_of_saved_models = os.listdir(saved_models_dir)
    list_of_models = []
    list_of_seeds = []
    if recompute == False:
        # We keep a list of pretrained models that include only those which
        # have not been trained
        for pm in os.listdir(path_to_pretrained_model):
            seed = int(pm.split('_')[0].replace('seed', ''))
            name_without_dirs = f'seed{seed}_epochs{epochs}.pt'
            if name_without_dirs not in list_of_saved_models:
                list_of_models.append(pm)
                list_of_seeds.append(seed)
        if len(list_of_models) == 0: 
            print(f"All specified models are already"
                " trained. Finishing process...")
            sys.exit(0)
        list_of_seeds.sort()

    # Sanitizing shard numbers
    if shard_number >= shards:
        print(f"You are asking for shard number {shard_number}, but there "
              f"are only {shards} shards. Finishing process... ")
        break

    # Loop over files
    for file in split(l=list_of_models, n=shards)[shard_number]: 
        """"We iterate over all the pretrained models in the specified route 
        above, and we extract the hyperparameters Nhid, seed, learning_rate 
        from the name of the file."""
        
        name_of_file = file
        path_to_file = '{}{}'.format(path_to_pretrained_model, file)
        seed = int(name_of_file.split('_')[0].replace('seed', ''))
        torch.manual_seed(seed)
        name_without_dirs = f'seed{seed}_epochs{epochs}.pt'
        # If the parameter 'recompute' is set to 'False' and the model
        # analyzed in the current iteration had been already trained, we
        # skip this model.
        if (recompute == False and 
            name_without_dirs in os.listdir(saved_models_dir)): 
            print(f'Skipping already trained model {name_of_file}...')
            continue             
        print(f'\nArch = {arch}, Neurons = {Nhid}, Actfun = {actfun}, ' \
                f'lr = {lr}, Alpha = {alpha}, Mu = {mom}, ' \
                  f'Seed = {seed} ({list_of_seeds.index(seed)}/{len(list_of_seeds)})')
        
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
        
        # We load our NN and optimizer to the CPU (or GPU)
        psi_ann = net_arch_map[arch](Nin, Nhid_prime, Nout, W1, 
                                     Ws2, B, W2, Wd2, actfun).to(device)
        psi_ann = torch.compile(psi_ann) if args.compile else psi_ann
        optimizer = getattr(torch.optim, optim)(params=psi_ann.parameters(),
                                        lr=lr,
                                        eps=epsilon,
                                        alpha=alpha,
                                        momentum=mom)
        stdict = torch.load(path_to_file)['optimizer_state_dict']
        psi_ann.load_state_dict(torch.load(path_to_file)['model_state_dict'])
        optimizer.load_state_dict(stdict)
        psi_ann.train()
        if show_arch:
            show_layers(psi_ann)
          
        ################### EPOCH LOOP ###################
        path_plot = f'saved_models/n3lo/{arch}/nhid{Nhid}/optim{optim}/' \
            f'{actfun}/lr{lr}/alpha{alpha}/mu{mom}/plots/' \
                f'seed{seed}_epochs{epochs}'
        
        # We store the energy data in lists for later plotting
        K_accum = []
        U_accum = []
        E_accum = []
        ks_accum = []
        kd_accum = []
        pd_accum = []
        
        start_time = time.time()
        # Training loop
        pbar = tqdm(range(epochs), desc="Training...", dynamic_ncols=True)
        for t in pbar:        
            (E, ann_s, ann_d, norm2, 
             K, U, ks, kd, pd) = train_loop(loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            model=psi_ann,
                                            train_data=Q_train, 
                                            q_2=q_2,
                                            arch=arch,
                                            integration=integration,
                                            w_i=w_i,
                                            q2_128=q2_128,
                                            wi_128=wi_128,
                                            V_ij=V_ij,
                                            psi_exact_s=psi_exact_s,
                                            psi_exact_s_norm=psi_exact_s_norm,
                                            psi_exact_d=psi_exact_d,
                                            psi_exact_d_norm=psi_exact_d_norm)
            K_accum.append(K.item())    
            U_accum.append(U.item())
            E_accum.append(E.item())
            ks_accum.append(ks.item())
            kd_accum.append(kd.item())
            pd_accum.append(pd.item())
            
            pbar.set_postfix_str(f"E = {E:.4f}, K={K:.4f}, U={U:.4f}")
            
            # Plotting
            if (((t+1) % leap) == 0 and periodic_plots) or t == epochs-1:
                show = show_last_plot if t == epochs - 1 else False
                minimisation_plots(x_axis=k,
                                   ann_s=ann_s,
                                   ann_d=ann_d,
                                   psi_exact_s_normalized=
                                   psi_exact_s_normalized,
                                   psi_exact_d_normalized=
                                   psi_exact_d_normalized,
                                   ks_accum=ks_accum,
                                   kd_accum=kd_accum,
                                   ks=ks,
                                   kd=kd,
                                   K_accum=K_accum,
                                   K=K,
                                   U=U,
                                   E=E,
                                   U_accum=U_accum,
                                   E_accum=E_accum,
                                   E_exact=E_exact,
                                   pd=pd,
                                   pd_accum=pd_accum,
                                   PD_exact=PD_exact,
                                   path_plot=path_plot,
                                   t=t,
                                   s=save_plot if t == epochs -1 else False,
                                   show=show) 
                
            pbar.update(1)

        pbar.close()
        print('\nModel trained!')
        print('Total execution time:  {:6.2f} seconds ' \
              '(run on {})'.format(time.time() - start_time_all, device))
        
        if save_model:
            path_model = f'saved_models/n3lo/{arch}/nhid{Nhid}/optim{optim}/' \
                f'{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/' \
                    f'seed{seed}_epochs{epochs}'
            full_path_model = f'{path_model}.pt'
            state_dict = {'model_state_dict':psi_ann.state_dict(),
                          'optimizer_state_dict':optimizer.state_dict()}
            torch.save(state_dict, full_path_model)
            print(f'Model saved in {full_path_model}')
        if save_plot: 
            full_path_plot = f'{path_plot}.pdf'
            print(f'Plot saved in {full_path_plot}')
        
print("\nAll done! :)")
