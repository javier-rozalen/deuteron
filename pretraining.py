# -*- coding: utf-8 -*-
#################### CONTENTS OF THE FILE ####################
"""
In this file we (pre)train an Artificial Neural Network to fit a given function
that resembles a wave function. We do this for different hyperparameters and 
network architectures. The trained models, as well as figures of the training
process, are stored under the folder 'saved_models/pretraining/'.
    
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
import os, sys
import torch, time, math
import numpy as np
from tqdm import tqdm
from itertools import product
import argparse

current = os.path.dirname(os.path.realpath(__file__))
os.chdir(current)

# My modules
import modules.integration as integration
import modules.neural_networks as neural_networks
from modules.plotters import pretraining_plots
from modules.aux_functions import dir_support, show_layers, split
from modules.aux_functions import pretrain_loop, strtobool
from modules.loss_functions import overlap

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="pretraining.py",
    usage="python3 %(prog)s [options]",
    description="Pretrains two neural networks to physical wave functions.",
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
    default=2000,
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
    "--seed-i", help="Seed number from which to start (default: 0)", default=0, type=int
)
parser.add_argument(
    "--seed-f", help="Seed number at which to end (default: 150)", default=150, type=int
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
    "--explore-hyperparams",
    help="Whether to train for the different combinations of hyperparameters "
    "specified or not (default: False). WARNING: this action will disable "
    "hyperparameter exploration even if lists of hyperparameters were given.",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
args = parser.parse_args()

##################### ADJUSTABLE PARAMETERS #####################
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
# The code handles the 30 <--> 32 conversion in hidden neurons
hidden_neurons = args.hidden_nodes
seed_from = args.seed_i
seed_to = args.seed_f
recompute = args.recompute

# Mesh parameters
q_max = 500
n_samples = 64  # Do not change this.
train_a = 0
train_b = q_max

# Training hyperparameters
actfuns = args.activations
optimizers = args.optimizers
learning_rates = args.lrs  # Use decimal notation
smoothing_constant = args.alphas
momentum = args.mus
epsilon = 1e-8

# Default hyperparam values
default_actfun = "Sigmoid"
default_optim = "RMSprop"
default_lr = 0.01
default_alpha = 0.9
default_momentum = 0.0

if not args.explore_hyperparams:
    actfuns = [default_actfun]
    optimizers = [default_optim]
    learning_rates = [default_lr]
    smoothing_constant = [default_alpha]
    momentum = [default_momentum]

################### DIRECTORY SUPPORT ###################
# Directory structure
path_steps_models = [
    "saved_models",
    "pretraining",
    "arch",
    "nhid",
    "optimizer",
    "actfun",
    "lr",
    "smoothing_constant",
    "momentum",
    "models/plots",
]

nsteps = range(len(path_steps_models))

################### MESH SET PREPARATION ###################
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

# Integration-specific
x_i_int = [torch.tensor(float(e)) for e in x]
w_i_int = [torch.tensor(float(e)) for e in w]

x_i = [torch.tensor(float(e) * 0.5 + 0.5) for e in x]  # q mesh from 0 to 1
# tangential q mesh from 0 to q_max
k = [
    (q_max / math.tan(x_i[-1] * math.pi / 2)) * math.tan(float(e) * math.pi / 2)
    for e in x_i
]
w_i = [torch.tensor(float(e) / 2) for e in w]  # GL weights
cos2 = [1 / (math.cos(float(x_i[i]) * math.pi / 2)) ** 2 for i in range(n_samples)]
p = (q_max / math.tan(x_i[-1] * math.pi / 2)) * math.pi / 2
w_i = [p * w_i[r] * cos2[r] for r in range(n_samples)]
w_i = torch.stack(w_i)
Q_train = torch.tensor(k)  # Momentum mesh
q_2 = Q_train**2  # Squared momentum mesh
Q_test = Q_train

################### TARGET FUNCTION ###################
psi_ansatz_s = torch.exp((-(1.5**2)) * Q_train**2 / 2)  # target function L=0
psi_ansatz_d = (Q_train**2) * torch.exp((-(1.5**2)) * Q_train**2 / 2)  # "" "" L=2

norm_s = integration.gl64(w_i, q_2 * (psi_ansatz_s**2))
norm_d = integration.gl64(w_i, q_2 * (psi_ansatz_d**2))

psi_ansatz_s_normalized = psi_ansatz_s / torch.sqrt(norm_s)
psi_ansatz_d_normalized = psi_ansatz_d / torch.sqrt(norm_d)

################### LOOP OVER SEEDS AND HYPERPARAMS ###################
start_time_all = time.time()
for arch, Nhid, optim, actfun, lr, alpha, mom in product(
    net_archs,
    hidden_neurons,
    optimizers,
    actfuns,
    learning_rates,
    smoothing_constant,
    momentum,
):
    Nhid = 32 if arch == "2sd" and Nhid == 30 else Nhid
    epochs = epochs_general

    # We restrict the pretraining to a specific hyperparameter configuration
    # Default hyperparams
    if not (
        (
            optim == default_optim
            and actfun == default_actfun
            and lr == default_lr
            and alpha == default_alpha
            and mom == default_momentum
        )
        or (
            Nhid == 60
            and
            # Variable lr
            (
                (
                    optim == default_optim
                    and actfun == default_actfun
                    and alpha == default_alpha
                    and mom == default_momentum
                )
                or
                # Variable actfun
                (
                    optim == default_optim
                    and lr == default_lr
                    and alpha == default_alpha
                    and mom == default_momentum
                )
                or
                # Variable alpha
                (
                    optim == default_optim
                    and actfun == default_actfun
                    and lr == default_lr
                    and mom == default_momentum
                )
                or
                # Variable momentum
                (
                    optim == default_optim
                    and actfun == default_actfun
                    and lr == default_lr
                    and alpha == default_alpha
                )
            )
        )
    ):
        # print('Skipping non-whitelisted hyperparam combination...')
        continue

    # Directory support
    for _ in nsteps:
        path_steps_models = [
            "saved_models",
            "pretraining",
            f"{arch}",
            f"nhid{Nhid}",
            f"optim{optim}",
            f"{actfun}",
            f"lr{lr}",
            f"alpha{alpha}",
            f"mu{mom}",
            "models",
        ]
        path_steps_plots = [
            "saved_models",
            "pretraining",
            f"{arch}",
            f"nhid{Nhid}",
            f"optim{optim}",
            f"{actfun}",
            f"lr{lr}",
            f"alpha{alpha}",
            f"mu{mom}",
            "plots",
        ]
        dir_support(path_steps_models)
        dir_support(path_steps_plots)

    # We reduce the list of seeds to iterate over to those seeds that have not
    # been already used
    l_seeds = list(range(seed_from, seed_to))
    if recompute == False:
        path_to_pretrained_models = "/".join(path_steps_models) + "/"
        for pm in os.listdir(path_to_pretrained_models):
            seed = int(pm.split("_")[0].replace("seed", ""))
            if seed in l_seeds:
                l_seeds.remove(seed)
    if len(l_seeds) == 0: 
        print(f"All specified seeds ({seed_from}-{seed_to}) were already used "
              "in previous pre-trainings. Finishing process...")
        sys.exit(0)
    l_seeds.sort()

    # We adjust the number of processes to the number of models
    if shard_number >= shards:
        print(f"You are asking for shard number {shard_number}, but there "
              f"are only {shards} shards. Finishing process... ")
        break

    # We iterate over the list of seeds we want to (pre)train
    for seed in split(l=l_seeds, n=shards)[shard_number]:
        saved_models_dir = "/".join(path_steps_models) + "/"
        name_without_dirs = f"seed{seed}_epochs{epochs}.pt"
        # If the current model has been already pretrained, we skip it
        if recompute == False and name_without_dirs in os.listdir(saved_models_dir):
            continue

        print(
            f"\nArch = {arch}, Neurons = {Nhid}, Optimizer = {optim},"
            f" Actfun = {actfun}, lr = {lr}, Alpha = {alpha}, Mu = {mom},"
            f" Seed = {seed}/{seed_to}"
        )
        torch.manual_seed(seed)
        path_model = (
            f"saved_models/pretraining/{arch}/nhid{Nhid}/"
            f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/"
            f"models/seed{seed}_epochs{epochs}"
        )
        path_plot = (
            f"saved_models/pretraining/{arch}/nhid{Nhid}/"
            f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/"
            f"plots/seed{seed}_epochs{epochs}"
        )

        # ANN Parameters
        if arch == "1sc":
            Nhid_prime = Nhid
        elif arch == "2sc" or arch == "1sd":
            Nhid_prime = int(Nhid / 2)
        elif arch == "2sd":
            Nhid_prime = int(Nhid / 4)
        Nin = 1
        Nout = 1 if arch == "2sd" else 2
        W1 = torch.rand(Nhid_prime, Nin, requires_grad=True) * (-1.0)
        B = torch.rand(Nhid_prime) * 2.0 - torch.tensor(1.0)
        W2 = torch.rand(Nout, Nhid_prime, requires_grad=True)
        Ws2 = torch.rand(1, Nhid_prime, requires_grad=True)
        Wd2 = torch.rand(1, Nhid_prime, requires_grad=True)

        # We load our psi_ann to the CPU (or GPU)
        net_arch_map = {
            "1sc": neural_networks.sc_1,
            "2sc": neural_networks.sc_2,
            "1sd": neural_networks.sd_1,
            "2sd": neural_networks.sd_2,
        }
        psi_ann = net_arch_map[arch](
            Nin=Nin,
            Nhid_prime=Nhid_prime,
            Nout=Nout,
            W1=W1,
            Ws2=Ws2,
            B=B,
            W2=W2,
            Wd2=Wd2,
            actfun=actfun,
        ).to(device)
        psi_ann = torch.compile(psi_ann) if args.compile else psi_ann

        # We define the loss function and the optimizer
        loss_fn = overlap
        optimizer = getattr(torch.optim, optim)(
            params=psi_ann.parameters(), lr=lr, eps=epsilon, alpha=alpha, momentum=mom
        )
        if show_arch:
            show_layers(psi_ann)

        ##################### EPOCH LOOP #####################
        # We store the energy data in lists for later plotting
        overlap_s, overlap_d = [], []

        start_time = time.time()
        for t in tqdm(range(epochs)):
            (psi_s_pred, psi_d_pred, k_s, k_d, loss) = pretrain_loop(
                model=psi_ann,
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
                psi_ansatz_d=psi_ansatz_d,
            )
            overlap_s.append(k_s.item())
            overlap_d.append(k_d.item())

            if ((t + 1) % leap) == 0 and periodic_plots:
                pretraining_plots(
                    x_axis=Q_test,
                    psi_s_pred=psi_s_pred,
                    psi_d_pred=psi_d_pred,
                    n_samples=n_samples,
                    psi_ansatz_s_normalized=psi_ansatz_s_normalized,
                    psi_ansatz_d_normalized=psi_ansatz_d_normalized,
                    overlap_s=overlap_s,
                    overlap_d=overlap_d,
                    path_plot=path_plot,
                    t=t,
                    s=save_plot if t == epochs - 1 else False,
                    show=False,
                )
            
        print("Model pretrained!")
        print(
            "Total execution time: {:6.2f} seconds (run on {})".format(
                time.time() - start_time_all, device
            )
        )

        full_path_model = "{}.pt".format(path_model)
        full_path_plot = "{}.pdf".format(path_plot)

        if save_model:
            state_dict = {
                "model_state_dict": psi_ann.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state_dict, full_path_model)
            print(f"Model saved in {full_path_model}")
        if save_plot:
            print(f"Plot saved in {full_path_plot}")

print("\nAll done! :)")
