# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This script computes the convergence rate and energy of trained models. The 
rate is computed as the number of models that have converged within the range 
E <= E_min MeV divided by the total number of models that have been trained. If
necessary, the data is stored under the 'error_data/' folder.
    
· In Section 'FILE SEARCHING' we specify a hyperparameter combinatioon and we 
search all trained models that match that criterium. We then count the number
of trained models and also the number of converged models, and we compute the
convergence rate. The data is then stored under the 'error_data/' folder. 
"""

############# IMPORTS #############
import os, sys, pathlib, statistics, math
from itertools import product
import argparse

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent_parent = os.path.dirname(parent)
sys.path.append(parent_parent)

# To simplify file creation using relative paths
os.chdir(current)

# My modules
from modules.aux_functions import strtobool


##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="convergence_rate.py",
    usage="python3 %(prog)s [options]",
    description="Computes convergence rates of the saved NN models.",
    epilog="Author: J Rozalén Sarmiento",
)

parser.add_argument(
    "--archs",
    help="List of NN architectures to train (default: 1sc 2sc 1sd 2sd)"
    "WARNING: changing this might entail further code changes to ensure proper"
    " functioning)",
    default=["1sc", "2sc", "1sd", "2sd"],
    nargs="*",
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
    "--min-E",
    help="Minimum allowed energy (default: -2.22)",
    default=-2.22,
    type=float
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
parser.add_argument(
    "--verbose",
    help="Whether to print detailed information about convergence or not "
    "(default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x))
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
args = parser.parse_args()


############# ADJUSTABLE PARAMETERS #############
# General parameters
net_archs = args.archs
save_data = args.save_data
E_min = args.min_E
print_info = args.verbose
n_models = 150

# Training hyperparameters
hidden_neurons = args.hidden_nodes
actfuns = args.activations
optimizers = args.optimizers
learning_rates = args.lrs  # Use decimal notation
smoothing_constant = args.alphas
momentum = args.mus

# Default hyperparameters
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

############# FILE SEARCHING #############
for arch, nhid, optim, actfun, lr, alpha, mom in product(
    net_archs,
    hidden_neurons,
    optimizers,
    actfuns,
    learning_rates,
    smoothing_constant,
    momentum,
):
    nhid = 32 if arch == "2sd" and nhid == 30 else nhid
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
            nhid == 60
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
        continue

    path_to_error_files = (
        f"../error_data/{arch}/nhid{nhid}/"
        f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/"
    )
    try:
        ###### ENERGIES ######
        path_to_filtered_files = (
            f"../error_data/{arch}/nhid{nhid}/"
            f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/filtered_runs/"
        )
        print("path to filtered files: ", os.listdir(path_to_filtered_files))
        energies = []
        E_top_errors = []
        E_bot_errors = []
        E_errors = [E_bot_errors, E_top_errors]
        for file in os.listdir(path_to_filtered_files):
            if (
                os.path.isfile(f"{path_to_filtered_files}{file}")
                and file.split(".")[1] == "txt"
                and file == "means_and_errors.txt"
            ):
                with open(f"{path_to_filtered_files}{file}", "r") as f:
                    for line in f.readlines():
                        energies.append(float(line.split(" ")[0]))
                        E_top_errors.append(float(line.split(" ")[1]))
                        E_bot_errors.append(float(line.split(" ")[2]))
                    f.close()
        if len(energies) > 1:
            mean_E = sum(energies) / len(energies)
            stdev_E = statistics.stdev(energies) / math.sqrt(len(energies))
            mean_E_top_errors = sum(E_top_errors) / len(E_top_errors)
            mean_E_bot_errors = sum(E_bot_errors) / len(E_bot_errors)
        else:
            print(
                f"energies has {len(energies)} elements, so we cannot"
                " compute the statistics..."
            )

        ###### CONVERGED MODELS ######
        n_converged = 0
        for file in os.listdir(path_to_error_files):
            if (
                os.path.isfile(f"{path_to_error_files}{file}")
                and file.split(".")[1] == "txt"
                and file == "means_and_errors.txt"
            ):
                with open(f"{path_to_error_files}{file}", "r") as f:
                    for line in f.readlines():
                        E = line.strip("\n").split(" ")[0]
                        if float(E) <= E_min:
                            n_converged += 1
                    f.close()

        ###### TRAINED MODELS ######
        path_to_trained_models = (
            f"../../saved_models/n3lo/{arch}/nhid{nhid}/"
            f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/"
        )
        n_trained = len(os.listdir(path_to_trained_models))
        if n_trained < n_models:
            print(
                f"Models at {path_to_trained_models} have only "
                f"{n_trained}/{n_models} trained models."
            )
        conv_rate = n_converged / n_trained

        if conv_rate > 1.0:
            raise Warning(
                "Oops! It seems that there are more converged "
                f"models at {path_to_error_files} than trained "
                f"models at {path_to_trained_models}"
            )

        if print_info:
            print(
                f"\nArch = {arch}, Neurons = {nhid}, Actfun = {actfun}, "
                f"lr = {lr}, Alpha = {alpha}, Mu = {mom}, "
                f"<E> = {mean_E}, E+ = ({stdev_E}, {mean_E_top_errors}), "
                f"E- = ({stdev_E}, {mean_E_bot_errors}), "
                "Rate = {:.2f}%".format(100 * conv_rate)
            )

        ###### DATA SAVING ######
        # We save the convergence rate data to a file
        if save_data:
            conv_error_file = path_to_error_files + "conv_rate.txt"
            # If the file we want to create already exists, we first delete it
            with open(conv_error_file, "w") as file:
                # n_converged n_trained conv_rate E E_top_stoch E_top_osc
                # E_bot_stoch E_bot_osc
                file.write(
                    f"{n_converged} {n_trained} {conv_rate} {mean_E} "
                    f"{stdev_E} {mean_E_top_errors} {stdev_E} "
                    f"{mean_E_bot_errors}"
                )
                file.close()
    except FileNotFoundError:
        print(
            "\nIt seems that there are no files under "
            f"{path_to_error_files}. Skipping this file..."
        )
