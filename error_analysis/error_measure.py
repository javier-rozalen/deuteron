# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This program runs the error analysis of the (fully) trained models. It computes 
the mean values of the energy, the fidelity and the probability of the D-state,
along with their associated errors. Further details can be found in the main
text of the article. All data is saved under the 'error_analysis/' folder.
    
· In Section 'MESH PREPARATION' we generate the momentum space mesh on which we 
train the networks.

· In Section 'BENCHMARK DATA FETCHING/COMPUTATION' we load/compute the wave 
functions against which we test our neural network models. 

· In Section 'LOOP OVER TRAINED MODELS' we perform a 300-epoch training on each
of the fully trained models. We also compute mean values and their associated
errors, and store this information under the 'error_data/' folder in the same
directory of this file. 
"""

######################## IMPORTS ########################
import pathlib, os, sys

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

os.chdir(current)

import torch, time, math
import numpy as np
from tqdm import tqdm
from itertools import product
import argparse

# My modules
from modules.physical_constants import hbar, mu
import modules.integration as integration
import modules.neural_networks as neural_networks
import modules.N3LO as N3LO
from modules.plotters import error_measure_plots
from modules.aux_functions import dir_support, split, show_layers, strtobool
from modules.aux_functions import train_loop, round_to_1, f, error_update
from modules.loss_functions import energy

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="error_measure.py",
    usage="python3 %(prog)s [options]",
    description="Computes the errors of NN trainings.",
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
    "--save-data",
    help="Whether to save the data or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "-e",
    help="Number of epochs for the pretraining (default: 300)",
    default=300,
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

######################## ADJUSTABLE PARAMETERS ########################
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
save_data = args.save_data
epochs = args.e
periodic_plots = args.periodic_plots
show_last_plot = args.plot
show_arch = args.show_arch
leap = epochs
recompute = args.recompute
print_stats_for_every_model = False

# Mesh parameters
q_max = 500
n_samples = 64  # Do not change this.
train_a = 0
train_b = q_max

# Adaptive plotting parameters
adaptive_lims = True
factor_E_sup = 1.000005
factor_E_inf = 1.00001
factor_k_sup = 1.001
factor_k_inf = 0.99997
factor_pd_sup = 1.001
factor_pd_inf = 0.999

# Training hyperparameters
hidden_neurons = args.hidden_nodes
actfuns = args.activations
optimizers = args.optimizers
learning_rates = args.lrs  # Use decimal notation
epsilon = 1e-8
smoothing_constant = args.alphas
momentum = args.mus

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

####################### MESH PREPARATION #######################
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

################ BENCHMARK DATA FETCHING/COMPUTATION ################
# N3LO potential
n3lo = N3LO.N3LO("../deuteron_data/2d_vNN_J1.dat", "../deuteron_data/wfk.dat")
V_ij = n3lo.getPotential()

# Target wavefunction (A.Rios, James W.T. Keeble)
psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)
psi_target_s_norm = integration.gl64(q_2 * (psi_target_s) ** 2, w_i)
psi_target_d_norm = integration.gl64(q_2 * (psi_target_d) ** 2, w_i)
psi_target_s_normalized = psi_target_s / torch.sqrt(
    psi_target_s_norm + psi_target_d_norm
)
psi_target_d_normalized = psi_target_d / torch.sqrt(
    psi_target_d_norm + psi_target_s_norm
)

# Vectors for the cost function
q2_128 = torch.cat((q_2, q_2))
wi_128 = torch.cat((w_i, w_i))

# Exact diagonalization wavefunction
psi_exact_s = []
psi_exact_d = []
Q_train1 = []
with open("../deuteron_data/wfk_exact_diagonalization.txt", "r") as file:
    c = 0
    for line in file.readlines():
        if c >= 2:
            psi_exact_s.append(torch.tensor(float(line.split(" ")[4])))
            psi_exact_d.append(torch.tensor(float(line.split(" ")[6])))
            Q_train1.append(torch.tensor(float(line.split(" ")[2])))
        c += 1

psi_exact_s = torch.stack(psi_exact_s)
psi_exact_d = torch.stack(psi_exact_d)
psi_exact_s_norm = integration.gl64(q_2 * (psi_exact_s) ** 2, w_i)
psi_exact_d_norm = integration.gl64(q_2 * (psi_exact_d) ** 2, w_i)
psi_exact_s_normalized = psi_exact_s / torch.sqrt(psi_exact_s_norm + psi_exact_d_norm)
psi_exact_d_normalized = psi_exact_d / torch.sqrt(psi_exact_s_norm + psi_exact_d_norm)


# Energy obtained with the exact diagonalization wf
def get_energy_pd():
    """Returns the energy and probability of D-state of the exact
    diagonalization wf"""

    psi_128 = torch.cat((psi_exact_s_normalized, psi_exact_d_normalized))
    y = psi_128 * q2_128 * wi_128
    U_exact = torch.matmul(y, torch.matmul(V_ij, y))
    K_exact = ((hbar**2) / mu) * torch.dot(wi_128, (psi_128 * q2_128) ** 2)
    E_exact = K_exact + U_exact
    PD_exact = 100.0 * psi_exact_d_norm / (psi_exact_s_norm + psi_exact_d_norm)
    return E_exact, PD_exact


E_exact, PD_exact = get_energy_pd()[0], get_energy_pd()[1]

####################### LOOP OVER TRAINED MODELS #######################
# We iterate over all the pretrained models
start_time_total = time.time()
if save_data:
    print("Data saving active...")
if shards > len(net_archs):
    nchunks = len(net_archs)
else:
    nchunks = shards
if shard_number >= nchunks:
    print("Error!")
for arch in split(l=net_archs, n=nchunks)[shard_number]:
    for Nhid, optim, actfun, lr, alpha, mom in product(
        hidden_neurons,
        optimizers,
        actfuns,
        learning_rates,
        smoothing_constant,
        momentum,
    ):
        Nhid = 32 if arch == "2sd" and Nhid == 30 else Nhid

        # We specify the hyperparam combination
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

        print("\nAnalyzing models with:")
        print(
            f"Arch = {arch}, Neurons = {Nhid}, Actfun = {actfun}, "
            f"lr = {lr}, Alpha = {alpha}, Mu = {mom}"
        )

        # We make a list of all models with the above hyperparams
        path_to_trained_models = (
            f"../saved_models/n3lo/{arch}/nhid{Nhid}/"
            f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/"
        )
        if not os.path.exists(path_to_trained_models):
            print(f"Path {path_to_trained_models} does not exist. Skipping...")
            continue
        list_of_trained_models = os.listdir(path_to_trained_models)
        sorting = lambda x: int(x.split("_")[0].replace("seed", ""))
        list_of_trained_models.sort(key=sorting)

        # Directory support
        path_steps_models = [
            "error_data",
            f"{arch}",
            f"nhid{Nhid}",
            f"optim{optim}",
            f"{actfun}",
            f"lr{lr}",
            f"alpha{alpha}",
            f"mu{mom}",
        ]
        dir_support(path_steps_models)

        potential_dir = "/".join(path_steps_models)
        final_model_name = potential_dir + "/means_and_errors.txt"
        final_model_name_without_dirs = "means_and_errors.txt"

        # We check whether the 'means_and_errors.txt' file already exists.
        # If it does, we collect the seed numbers of the analyzed models.
        # If it does not, we create the file.
        l_analyzed_seeds = []
        if recompute == False and os.path.isfile(final_model_name):
            with open(final_model_name, "r") as mr:
                for line in mr.readlines():
                    seedi = int(line.split(" ")[-2])
                    l_analyzed_seeds.append(seedi)
                mr.close()
        else:
            with open(final_model_name, "w") as mr:
                mr.close()

        for model in tqdm(list_of_trained_models):
            """We iterate over all trained models."""

            seed = int(model.split("_")[0].replace("seed", ""))

            if recompute == False and seed in l_analyzed_seeds:
                # print(f'Skipping seed {seed} of {final_model_name}...')
                continue

            # Single-layer specific params
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
                Nin, Nhid_prime, Nout, W1, Ws2, B, W2, Wd2, actfun
            ).to(device)
            psi_ann.load_state_dict(
                torch.load(f"{path_to_trained_models}" f"{model}")["model_state_dict"]
            )

            if show_arch:
                show_layers(psi_ann)

            ############### TRAINING AND TESTING ###############
            # Training parameters
            loss_fn = energy
            optimizer = torch.optim.RMSprop(
                params=psi_ann.parameters(),
                lr=lr,
                eps=epsilon,
                alpha=alpha,
                momentum=mom,
            )
            stdict = f"{path_to_trained_models}{model}"
            optimizer.load_state_dict(torch.load(stdict)["optimizer_state_dict"])

            ############### EPOCH LOOP ###############
            # We store the energy data in lists so as to plot it
            K_accum = []
            U_accum = []
            E_accum = []
            ks_accum = []
            kd_accum = []
            pd_accum = []

            # Error analysis parameters
            mean_E_accum = []
            E_top_accum = []
            E_bot_accum = []
            mean_ks_accum = []
            mean_kd_accum = []
            ks_top_accum = []
            ks_bot_accum = []
            kd_top_accum = []
            kd_bot_accum = []
            mean_PD_accum = []
            pd_top_accum = []
            pd_bot_accum = []

            mean_E = 0
            mean_E_top = 0
            mean_E_bot = 0
            mean_ks = 0
            mean_kd = 0
            mean_ks_top = 0
            mean_ks_bot = 0
            mean_kd_top = 0
            mean_kd_bot = 0
            mean_PD = 0
            mean_pd_top = 0
            mean_pd_bot = 0

            start_time = time.time()
            for t in range(epochs):
                vars_ = train_loop(
                    loss_fn=loss_fn,
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
                    psi_exact_d_norm=psi_exact_d_norm,
                )
                E, ann_s, ann_d, norm2, K, U, ks, kd, pd = vars_
                K_accum.append(K.item())
                U_accum.append(U.item())
                E_accum.append(E.item())
                ks_accum.append(ks.item())
                kd_accum.append(kd.item())
                pd_accum.append(pd.item())

                # We update the different error quantities with the information
                # from the current iteration
                if t > 3:
                    vars_ = error_update(
                        E_accum=E_accum,
                        mean_E_accum=mean_E_accum,
                        E_top_accum=E_top_accum,
                        E_bot_accum=E_bot_accum,
                        mean_E_top=mean_E_top,
                        mean_E_bot=mean_E_bot,
                        mean_ks_top=mean_ks_top,
                        mean_ks_bot=mean_ks_bot,
                        mean_kd_top=mean_kd_top,
                        mean_kd_bot=mean_kd_bot,
                        mean_pd_top=mean_pd_top,
                        mean_pd_bot=mean_pd_bot,
                        mean_ks=mean_ks,
                        ks_accum=ks_accum,
                        mean_ks_accum=mean_ks_accum,
                        ks_top_accum=ks_top_accum,
                        ks_bot_accum=ks_bot_accum,
                        mean_kd=mean_kd,
                        kd_accum=kd_accum,
                        mean_kd_accum=mean_kd_accum,
                        kd_top_accum=kd_top_accum,
                        kd_bot_accum=kd_bot_accum,
                        mean_PD=mean_PD,
                        pd_accum=pd_accum,
                        mean_PD_accum=mean_PD_accum,
                        pd_top_accum=pd_top_accum,
                        pd_bot_accum=pd_bot_accum,
                    )
                    (
                        mean_E,
                        mean_E_accum,
                        E_top_accum,
                        E_bot_accum,
                        mean_E_top,
                        mean_E_bot,
                        mean_ks,
                        mean_ks_accum,
                        ks_top_accum,
                        ks_bot_accum,
                        mean_ks_top,
                        mean_ks_bot,
                        mean_kd,
                        mean_kd_accum,
                        kd_top_accum,
                        kd_bot_accum,
                        mean_kd_top,
                        mean_kd_bot,
                        mean_PD,
                        mean_PD_accum,
                        pd_top_accum,
                        pd_bot_accum,
                        mean_pd_top,
                        mean_pd_bot,
                        break_,
                    ) = vars_

                    if break_:
                        break

                if mean_E <= 0.0:
                    # Plotting
                    if (((t + 1) % leap) == 0 and periodic_plots) or (
                        t == epochs - 1 and show_last_plot
                    ):
                        error_measure_plots(
                            adaptive_lims=adaptive_lims,
                            ks_accum=ks_accum,
                            factor_k_sup=factor_k_sup,
                            factor_k_inf=factor_k_inf,
                            kd_accum=kd_accum,
                            ks=ks,
                            kd=kd,
                            mean_ks=mean_ks,
                            mean_ks_top=mean_ks_top,
                            mean_ks_bot=mean_ks_bot,
                            mean_kd=mean_kd,
                            mean_kd_bot=mean_kd_bot,
                            mean_kd_top=mean_kd_top,
                            factor_E_sup=factor_E_sup,
                            factor_E_inf=factor_E_inf,
                            E_accum=E_accum,
                            E=E,
                            E_exact=E_exact,
                            mean_E=mean_E,
                            mean_E_top=mean_E_top,
                            mean_E_bot=mean_E_bot,
                            factor_pd_sup=factor_pd_sup,
                            factor_pd_inf=factor_pd_inf,
                            pd_accum=pd_accum,
                            pd=pd,
                            PD_exact=PD_exact,
                            mean_PD=mean_PD,
                            mean_pd_top=mean_pd_top,
                            mean_pd_bot=mean_pd_bot,
                        )

            if mean_E <= 0:
                # Once post-training evolution finishes, we compute global values
                (
                    mean_E_top,
                    mean_E_bot,
                    mean_ks_top,
                    mean_ks_bot,
                    mean_kd_top,
                    mean_kd_bot,
                    mean_pd_top,
                    mean_pd_bot,
                ) = f(
                    E_accum,
                    ks_accum,
                    kd_accum,
                    pd_accum,
                    print_stats_for_every_model,
                    mean_E,
                    mean_ks,
                    mean_kd,
                    mean_PD,
                )

            # We save the data in a file
            if save_data:
                with open(final_model_name, "a") as file:
                    try:
                        # E, E+, E-, Ks, Ks+, Ks-, Pd, Pd+, Pd-, Seed
                        E_plus = round_to_1(abs(abs(mean_E_top) - abs(mean_E)))
                        E_minus = round_to_1(abs(abs(mean_E_bot) - abs(mean_E)))
                        Ks_plus = round_to_1(abs(abs(mean_ks_top) - abs(mean_ks)))
                        Ks_minus = round_to_1(abs(abs(mean_ks_bot) - abs(mean_ks)))
                        Kd_plus = round_to_1(abs(abs(mean_kd_top) - abs(mean_kd)))
                        Kd_minus = round_to_1(abs(abs(mean_kd_bot) - abs(mean_kd)))
                        Pd_plus = round_to_1(abs(abs(mean_pd_top) - abs(mean_PD)))
                        Pd_minus = round_to_1(abs(abs(mean_pd_bot) - abs(mean_PD)))
                        file.write(
                            str(mean_E)
                            + " "
                            + str(E_plus)
                            + " "
                            + str(E_minus)
                            + " "
                            + str(mean_ks)
                            + " "
                            + str(Ks_plus)
                            + " "
                            + str(Ks_minus)
                            + " "
                            + str(mean_kd)
                            + " "
                            + str(Kd_plus)
                            + " "
                            + str(Kd_minus)
                            + " "
                            + str(mean_PD)
                            + " "
                            + str(Pd_plus)
                            + " "
                            + str(Pd_minus)
                            + " "
                            + str(seed)
                            + " \n"
                        )
                        file.close()
                    except:
                        file.write(str(mean_E) + " " + str(seed) + " \n")

end_time_total = time.time()
print("\nAll done! :)")
print(
    "\nTotal execution time: "
    "{:6.2f} seconds (run on {})".format(end_time_total - start_time_total, device)
)
