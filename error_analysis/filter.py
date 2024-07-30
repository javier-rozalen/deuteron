# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This program filters the data produced by the script 'error_measure.py', i.e., 
it creates a folder called 'filtered_runs/' in the same location as the error
files, where it will store a maximum of 20 converged models. The convergence
criteria is defined via the variable 'E_min'. 

· In Section 'ADJUSTABLE PARAMETERS' we can change the values of the variables.
Below we explain what each of these variables do.
    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    E_min --> float, (mean) Energy below which we stop accepting runs.
    good_runs_min --> int, Minimum number of good (filtered) runs we wish to 
                    end up with. If this is not the case, a message is printed 
                    via console with additional information.
    hidden_neurons_list --> list, Number of neurons list that we wish to keep. 
                        (Note that we may have trained models with a different 
                         number of nhid)
    seed --> int, Random seed used for shuffling.
    
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

· In Section 'FILES, DIRECTORIES' we identify all error files that match our 
specific hyperparameter combination and we create the 'filtered_runs/ ' folder
for such files. We copy all error files to the 'filtered_runs/' folder. 

· In Section 'FILTERING, COUNTING' we identify the converged files among those
under the 'filtered_runs/' folders, and we remove the rest. If there are more
than 20 converged files, we keep a random subset of 20 files. 

· In Section 'PRINGING' we print out to the console all the feedback about the 
number of trained, analyzed and converged models. 
"""

############# IMPORTS #############
import pathlib, os, sys, shutil, random
import argparse

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

os.chdir(current)

from itertools import product

# My modules
from modules.aux_functions import dir_support, strtobool

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="pretraining.py",
    usage="python3 %(prog)s [options]",
    description="Pretrains two neural networks to physical wave functions.",
    epilog="Author: J Rozalén Sarmiento",
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
    "--E-min",
    help="Upper energy threshold (in MeV) above which models are considered not "
    "converged (default: -2.22)",
    default=-2.22,
    type=float
)
parser.add_argument(
    "--n-models",
    help="Number of trained models to consider (default: 150)",
    default=150,
    type=int
)
parser.add_argument(
    "--good-runs-min",
    help="Minimum number of converged models allowed (default: 20)",
    default=20,
    type=int
)
args = parser.parse_args()

############# ADJUSTABLE PARAMETERS #############
# General parameters
net_archs = ["1sc", "2sc", "1sd", "2sd"]
E_min = args.E_min
n_models = args.n_models
good_runs_min = args.good_runs_min
seed = 2

# Training hyperparameters
hidden_neurons = [20, 30, 40, 60, 80, 100]
optimizers = ["RMSprop"]
actfuns = ["Sigmoid", "Softplus", "ReLU"]
learning_rates = [0.005, 0.01, 0.05]  # Use decimal notation
epsilon = 1e-8
smoothing_constant = [0.7, 0.8, 0.9]
momentum = [0.0, 0.9]

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

############# FILES, DIRECTORIES #############
random.seed(seed)
l_few_trained_models = []
l_few_analyzed_models = []
l_few_good_runs = []
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

    # At each hyperparams configuration we start from the root
    # directory where this script is found.
    os.chdir(current)
    # First we check that there are as many trained models as the variable
    # n_models specifies.
    path_to_trained_models = (
        f"../saved_models/n3lo/{arch}/nhid{nhid}/"
        f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/models/"
    )
    n_trained = len(os.listdir(path_to_trained_models))
    print(f"Files at {path_to_trained_models} have {n_trained} trained models.")
    if n_trained < n_models:
        l_few_trained_models.append(path_to_trained_models)

    path_to_error_files = (
        f"error_data/{arch}/nhid{nhid}/"
        f"optim{optim}/{actfun}/lr{lr}/alpha{alpha}/mu{mom}/"
    )
    os.chdir(path_to_error_files)

    """
    We prepare a directory called 'filtered_runs' to which we 
    copy all error files with 'nhid' in 'hidden_neurons_list'.
    """

    all_files = os.listdir()
    list_of_error_files = []
    files_not_wanted = []

    # We select the error files among all the files in this directory
    for file in all_files:
        if len(file.split(".")) > 1 and file == "means_and_errors.txt":
            list_of_error_files.append(file)
        else:
            files_not_wanted.append(file)

    # If the filtered_runs folder does not exist, we create it
    dir_support(["filtered_runs"])

    # We copy the error files into the filtered_runs folder
    for file in list_of_error_files:
        shutil.copy(file, "filtered_runs/")

    os.chdir("filtered_runs")

    # We remove those files with nhid not in hidden_neurons_list
    for file in os.listdir():
        if file in files_not_wanted:
            os.remove(file)

    ############# FILTERING, COUNTING #############
    """
    We create new error files that contain only good_runs_min good runs tops, 
    with all single energies being < E_min. 
    """

    # We go through each of the files keeping only the lines with E < E_min MeV
    deleted_Es = []
    incomplete_files = 0
    if len(os.listdir()) > 0:
        for file in os.listdir():
            if file == "means_and_errors.txt":
                good_runs = 0

                # We store the data of the file in a list
                with open(file, "r") as f:
                    lines = f.readlines()
                    num_lines = len([l for l in lines if l.strip(" \n") != ""])
                    if num_lines < n_models:
                        l_few_analyzed_models.append(path_to_error_files)
                        l_few_analyzed_models.append(num_lines)

                # We destroy the file, writing only the lines that fulfil our
                # criteria
                with open(file, "w") as f:
                    for line in lines:
                        E = line.strip("\n").split(" ")[0]
                        if float(E) <= E_min:
                            f.write(line)
                            good_runs += 1
                        else:
                            deleted_Es.append(E)
                    # We print a warning if the file has insufficient data
                    if good_runs < good_runs_min:
                        l_few_good_runs.append(path_to_error_files)
                        l_few_good_runs.append(good_runs)
                        incomplete_files += 1

                # We shuffle the lines of the files with
                # good_runs > good_runs_min
                if good_runs > good_runs_min:
                    with open(file, "r") as f:
                        lines = f.readlines()
                        random.shuffle(lines)
                        shuffled_data = []
                        for line in lines:
                            if len(shuffled_data) < good_runs_min:
                                shuffled_data.append(line)
                            else:
                                break

                    # We open the same file and write only the first
                    # good_runs_min lines, which is a random subset of the
                    # initial good_runs
                    with open(file, "w") as f:
                        for line in shuffled_data:
                            f.write(line)

    else:
        print("No error file found here!. The model is:")
        print(
            f"{arch} nhid{nhid} optim{optim} "
            f"actfun{actfun} lr{lr} alpha{alpha} mu{mom}"
        )

############# PRINTING #############
os.chdir(current)
print('\nMODELS TRAINED WITH "error_measure.py"')
print("------------------------------------")
for path in l_few_trained_models:
    print(
        f"Files at {path} only have "
        f"{len(os.listdir(path))}/{n_models} trained models."
    )
if len(l_few_trained_models) == 0:
    print(
        f"There are {n_models}/{n_models} trained models for the specified "
        "hyperparameter combination."
    )

print('\nMODELS ANALYZED WITH "deuteron.py"')
print("------------------------------------")
for path in l_few_analyzed_models:
    if type(path) == str:
        print(
            f"File at {path} only has "
            f"{l_few_analyzed_models[l_few_analyzed_models.index(path) + 1]}/"
            f"{n_models} analyzed models."
        )
if len(l_few_analyzed_models) == 0:
    print(
        f"There are {n_models}/{n_models} analyzed models for the specified"
        " hyperparameter combination."
    )

print("\nNUMBER OF CONVERGED MODELS")
print("------------------------------------")
for path in l_few_good_runs:
    if type(path) == str:
        print(
            f"File at {path} only has "
            f"{l_few_good_runs[l_few_good_runs.index(path) + 1]}/"
            f"{good_runs_min} good runs."
        )
if len(l_few_good_runs) == 0:
    print(
        f"All models have {good_runs_min}/{good_runs_min} converged models "
        "for the specified hyperparameter combination."
    )
