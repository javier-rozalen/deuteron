# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This program generates the oscillation error plot corresponding to FIG. 3 in 
the paper. The plot is saved under the 'error_analysis/' folder.

· In Section 'ADJUSTABLE PARAMETERS' we can change the values of the variables.
An extensive explanation of what each variable does is given below.
    
    MESH PARAMETERS 
    -----------------------------------
    q_max --> float, Maximum value of the momentum axis
    n_samples --> int, Number of mesh points. DO NOT CHANGE THE DEFAULT VALUE.
    train_a --> float, Lower limit of momentum
    train_b --> float, Upper limit of momentum. It is set to q_max by default.
    
    ADAPTIVE PLOTTING PARAMETERS
    -----------------------------------
    adaptive_lims --> Boolean, Sets whether the plot limits are adaptive or 
                    not.
    factor_E_sup --> Float, Scale factor of E that sets the E plot upper limit.
    factor_E_inf --> Float, Scale factor of E thta sets the E plot lower limit.
    factor_k_sup --> Float, Idem. 
    factor_k_inf --> Float, Idem.
    factor_pd_sup --> Float, Idem.
    factor_pd_inf --> Float, Idem.    
    
· In Section 'MESH PREPARATION' we generate the momentum space mesh on which we 
train the networks.

· In Section 'BENCHMARK DATA FETCHING/COMPUTATION' we load/compute the wave 
functions against which we test our neural network models. 

· In Section 'LOOP OVER TRAINED MODELS' we perform a 300-epoch training on each
of the fully trained models. We also compute mean values and their associated
errors, and store this information under the 'error_data/' folder in the same
directory of this file. 
"""

############# IMPORTS #############
import pathlib, os, sys

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)

import torch, time, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import argparse

# My modules
from modules.physical_constants import hbar, mu
import modules.integration as integration
import modules.neural_networks as neural_networks
import modules.N3LO as N3LO
from modules.aux_functions import show_layers, train_loop, f, error_update, dir_support, strtobool
from modules.loss_functions import energy

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="oscillation_error_plot.py",
    usage="python3 %(prog)s [options]",
    description="Generates a plot where the oscillation errors are shown.",
    epilog="Author: J Rozalén Sarmiento",
)
parser.add_argument(
    "--dev",
    help="Hardware device on which the code will run (default: cpu)",
    default="cpu",
    choices=["cpu", "gpu"],
    type=str
)
parser.add_argument(
    "-e",
    help="Number of epochs for the pretraining (default: 300)",
    default=300,
    type=int
)
parser.add_argument(
    "--save-plot",
    help="Whether to save the plot or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--show-pics",
    help="Whether to show plots as they are created or not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--periodic-plots",
    help="Whether to show plots periodically as models are being trained or "
    "not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--show-arch",
    help="Whether to show the NN architecture or not "
    "(default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--stats-forall",
    help="Whether to print statistics for every model or not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--adaptive-lims",
    help="Whether to make plot limits adaptive or not (default: False)",
    default=False,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
)
parser.add_argument(
    "--arch",
    help="Architecture of the NN model to use (default: 1sc)",
    default="1sc",
    choices=["1sc", "2sc", "1sd", "2sd"],
    type=str
)
parser.add_argument(
    "--hidden-nodes",
    help="Number of hidden nodes to use (default: 100)",
    default=100,
    choices=[20, 30, 32, 40, 60, 80, 100],
    type=int
)
parser.add_argument(
    "--activation",
    help="Activation function to use (default: Sigmoid)",
    default="Sigmoid",
    choices=["Sigmoid", "ReLU", "Softplus"],
    type=str
)
parser.add_argument(
    "--lr",
    help="Learning rate to use (default: 0.01)",
    default=0.01,
    choices=[0.05, 0.01, 0.1],
    type=float
)
parser.add_argument(
    "--alpha",
    help="Smoothing constant to use (default: 0.9)",
    default=0.9,
    choices=[0.7, 0.8, 0.9],
    type=float
)
parser.add_argument(
    "--momentum",
    help="Momentum to use (default: 0.0)",
    default=0.0,
    choices=[0.0, 0.9],
    type=float
)
parser.add_argument(
    "--seed",
    help="Seed to use (default: 1)",
    default=1,
    type=int
)
args = parser.parse_args()


############# ADJUSTABLE PARAMETERS #############
# General parameters
device = args.dev
if device == "gpu":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("There was an error trying to use the GPU. Falling back to CPU...")
        device = "cpu"
epochs = args.e
save_plot = args.save_plot
show_pics = args.show_pics
periodic_plots = args.periodic_plots
show_arch = args.show_arch
print_stats_for_every_model = args.stats_forall
leap = epochs

# Mesh parameters
q_max = 500
n_samples = 64  # Do not change this.
train_a = 0
train_b = q_max

# Adaptive plotting parameters
adaptive_lims = args.adaptive_lims
factor_E_sup = 1.000005
factor_E_inf = 1.00001
factor_k_sup = 1.00
factor_k_inf = 0.99997
path_of_plot = "../saved_plots/oscillation_error_plot.pdf"

# Hyperparameters
arch = args.arch
nhid = args.hidden_nodes
actfun = args.activation
lr = args.lr
alpha = args.alpha
mom = args.momentum
seed = args.seed
epsilon = 1e-8
optim = "RMSprop"

############# MESH SET PREPARATION #############
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

############# BENCHMARK DATA FETCHING/COMPUTATION #############
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


####################### PLOTTING FUNCTION #######################
def pic_creator(
    adaptive_lims,
    factor_k_inf,
    factor_k_sup,
    factor_E_inf,
    factor_E_sup,
    ks_accum,
    kd_accum,
    ks,
    kd,
    mean_ks,
    mean_ks_top,
    mean_ks_bot,
    mean_kd,
    mean_kd_bot,
    mean_kd_top,
    E_accum,
    E,
    E_exact,
    mean_E,
    mean_E_top,
    mean_E_bot,
    s,
    path_of_plot,
):
    linestyles = ["dashed", "dotted", "dashdot"]
    colors = ["green", "tab:orange", "purple"]
    labels_F = ["Mean $F^D$", "$F^D_-$", "$F^D_+$"]
    labels_E = ["Mean $E$", "$E_-$", "$E_+$"]
    data_F = [mean_kd, mean_kd_bot, mean_kd_top, mean_kd_bot]
    data_E = [mean_E, mean_E_bot, mean_E_top]
    if adaptive_lims == False:
        factor_k_inf, factor_k_sup, factor_E_inf, factor_E_sup = 1.0, 1.0, 1.0, 1.0

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 7), sharex="col")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.15)

    # Overlap
    axF = ax[0]
    axF.set_ylabel("Fidelity of D-state", fontsize=18, labelpad=15)
    axF.set_ylim(0.999955 * factor_k_inf, 1.0 * factor_k_sup)
    axF.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    axF.tick_params(axis="both", which="both", direction="in", labelsize=16, pad=8)
    axF.tick_params(axis="both", which="major", width=1.7, length=4.5)
    axF.yaxis.set_minor_locator(MultipleLocator(0.000005))

    axF.plot(
        torch.linspace(0, len(kd_accum), len(kd_accum)).numpy(),
        kd_accum,
        label="$F^D$",
        linestyle="solid",
    )
    for data, color, linestyle, label in zip(data_F, colors, linestyles, labels_F):
        axF.axhline(y=data, color=color, linestyle=linestyle, label=label)

    # Energy
    axE = ax[1]
    axE.set_xlabel("Epoch", fontsize=18)
    axE.set_ylabel("E (MeV)", fontsize=18, labelpad=15)
    axE.set_xlim(0, 300)
    axE.set_ylim(-2.226 * factor_E_inf, -2.22515 * factor_E_sup)  # fixed ylim
    axE.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axE.tick_params(axis="both", which="both", direction="in", labelsize=16, pad=8)
    axE.tick_params(axis="both", which="major", width=1.7, length=4.5)
    axE.yaxis.set_minor_locator(AutoMinorLocator())
    axE.xaxis.set_minor_locator(AutoMinorLocator())

    axE.plot(
        torch.linspace(0, len(E_accum), len(E_accum)).numpy(), E_accum, label="$E$"
    )

    for data, color, linestyle, label in zip(data_E, colors, linestyles, labels_E):
        axE.axhline(y=data, color=color, linestyle=linestyle, label=label)

    axF.legend(loc="lower center", ncol=5, fancybox=True, fontsize=15)
    axE.legend(loc="lower center", ncol=4, fancybox=True, fontsize=15)

    if s:
        dir_support(["../saved_plots/"])
        plt.savefig(path_of_plot, format="pdf", bbox_inches="tight")
        print(f"Figure correctly saved at {path_of_plot}.")

    if show_pics:
        plt.pause(0.001)


####################### ERROR COMPUTATION #######################
start_time_total = time.time()
model = (
    f"../saved_models/n3lo/{arch}/nhid{nhid}/"
    f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/"
    f"seed{seed}_epochs250000.pt"
)
if not os.path.isfile(model):
    print(f"The file {model} does not exist. Please make "
          "sure to select valid hyperparameters. Exitting...")
    sys.exit(0)

# ANN Parameters
if arch == "1sc":
    Nhid_prime = nhid
elif arch == "2sc" or arch == "1sd":
    Nhid_prime = int(nhid / 2)
elif arch == "2sd":
    Nhid_prime = int(nhid / 4)

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
psi_ann = net_arch_map[arch](Nin, Nhid_prime, Nout, W1, Ws2, B, W2, Wd2, actfun).to(
    device
)
psi_ann.load_state_dict(torch.load(model)["model_state_dict"])

if show_arch:
    show_layers(psi_ann)

####################### TRAINING AND TESTING #######################
# Training parameters
loss_fn = energy
optimizer = getattr(torch.optim, optim)(
    params=psi_ann.parameters(), lr=lr, eps=epsilon, alpha=alpha, momentum=mom
)
optimizer.load_state_dict(torch.load(model)["optimizer_state_dict"])

####################### EPOCH LOOP #######################
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
for t in tqdm(range(epochs)):
    (E, ann_s, ann_d, norm2, K, U, ks, kd, pd) = train_loop(
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

    K_accum.append(K.item())
    U_accum.append(U.item())
    E_accum.append(E.item())
    ks_accum.append(ks.item())
    kd_accum.append(kd.item())
    pd_accum.append(pd.item())

    # We update the different error quantities with the information
    # from the current iteration
    if t > 3:
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
        ) = error_update(
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
        if break_:
            break

    if mean_E <= 0.0:
        # Plotting
        if (((t + 1) % leap) == 0 and periodic_plots) or (t == epochs - 1):
            if t == epochs - 1:
                # Once post-training evolution finishes, we compute global
                # values
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
            pic_creator(
                adaptive_lims=adaptive_lims,
                factor_k_inf=factor_k_inf,
                factor_k_sup=factor_k_sup,
                factor_E_inf=factor_E_inf,
                factor_E_sup=factor_E_sup,
                ks_accum=ks_accum,
                kd_accum=kd_accum,
                ks=ks,
                kd=kd,
                mean_ks=mean_ks,
                mean_ks_top=mean_ks_top,
                mean_ks_bot=mean_ks_bot,
                mean_kd=mean_kd,
                mean_kd_bot=mean_kd_bot,
                mean_kd_top=mean_kd_top,
                E_accum=E_accum,
                E=E,
                E_exact=E_exact,
                mean_E=mean_E,
                mean_E_top=mean_E_top,
                mean_E_bot=mean_E_bot,
                s=save_plot if t == epochs - 1 else False,
                path_of_plot=path_of_plot,
            )
