# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This file generates FIG. 4 in the paper. The plot generated is stored under the
'saved_plots/' folder.

· In Section 'ADJUSTABLE PARAMETERS' we can change the values of the variables.
An explanation of what each variable does is given below.

    GENERAL PARAMETERS
    -----------------------------------
    net_archs --> List, Can contain any combination of the elements '1sc', 
                '2sc', '1sd', '2sd'. Each element is a network architecture.
    hidden_neurons --> List, Contains different numbers of hidden neurons.
    save_plot --> Boolean, Sets whether the generated plot is saved or not.
    path_of_plot --> str, relative_directory/name_of_plot.pdf
    show_osc_error --> Boolean, Whether the oscillation error areas are shown
                    in the plot or not.
    show_stoch_error --> Boolean, Whether the stochastic error areas are shown
                    in the plot or not.
                    
· In Section 'FURTHER ERROR COMPUTATIONS' we first go over all the error files 
generated with 'error_measure.py', and we compute further necessary errors. For 
each network architecture, we store all this information in a single file named 
'joint_graph.txt', which contains all the necessary information to generate the 
joint graph. Then, the data of this file is appended to the 
'master_plot_data_?.txt' files, each of which contains data from one network 
architecture.

· In Section 'DATA PREPARATION FOR PLOTTING' we read the files created in the 
previous section and load dictionaries with it. 

· In Section 'PLOTTING' the dictionaries in the previous section are read and 
the data is finally plotted.
"""

############# IMPORTS #############
import matplotlib.pyplot as plt
import math, os, statistics, sys
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import MultipleLocator
import argparse

# PYTHONPATH additions
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

# Allows running the file from different directories
os.chdir(current)

# My modules
from modules.aux_functions import dir_support, strtobool

##################### ARGUMENTS #####################
parser = argparse.ArgumentParser(
    prog="filter.py",
    usage="python3 %(prog)s [options]",
    description="Filters the trained models and looks for the converged ones.",
    epilog="Author: J Rozalén Sarmiento",
)
parser.add_argument(
    "--save-plot",
    help="Whether to save the plo or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--show-osc-error",
    help="Whether to show the oscillation error in the plot or not (default: True)",
    default=True,
    choices=[True, False],
    type=lambda x: bool(strtobool(x)),
)
parser.add_argument(
    "--show-stoch-error",
    help="Whether to show the stochastic error in the plot or not (default: True)",
    default=True,
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
    "--hidden-nodes",
    help="List of hidden node numbers to use (default: 20 30 40 60 80 100)",
    default=[20, 30, 40, 60, 80, 100],
    nargs="*",
    type=int
)
args = parser.parse_args()


############# ADJUSTABLE PARAMETERS #############
# General parameters
net_archs = args.archs
hidden_neurons = args.hidden_nodes
save_plot = args.save_plot
show_osc_error = args.show_osc_error
show_stoch_error = args.show_stoch_error

############# MISC. #############
dir_support(["plotting_data"])
path_of_plot = "../saved_plots/master_plot.pdf"

############# FURTHER ERROR COMPUTATIONS #############
for arch in net_archs:
    # We create the file 'master_plot_data_{arch}.txt'
    with open(f"plotting_data/master_plot_data_{arch}.txt", "w") as global_file:
        global_file.close()

    # We restrict to the default hyperparam values, letting only Nhid run
    for Nhid in hidden_neurons:
        Nhid = 32 if arch == "2sd" and Nhid == 30 else Nhid
        # We look for the (filtered) error files.
        path_to_error_files = (
            f"../error_analysis/error_data/"
            f"{arch}/nhid{Nhid}/optimRMSprop/Sigmoid/"
            f"lr0.01/alpha0.9/mu0.0/filtered_runs/"
        )

        # We check wether 'joint_graph_file.txt' exists and we delete it if it
        # does. Then we create a blank instance of the file.
        joint_graph_file = "{}joint_graph.txt".format(path_to_error_files)
        if os.path.isfile(joint_graph_file):
            os.remove(joint_graph_file)
        with open(joint_graph_file, "a") as f:
            f.close()

        all_files = os.listdir(path_to_error_files)
        list_of_error_files = [
            file for file in all_files if file == "means_and_errors.txt"
        ]

        for file in list_of_error_files:

            ################## ERROR DATA FETCHING ##################
            filename = "{}{}".format(path_to_error_files, file)

            test_number = []

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

            # We fill the lists above with the parameters computed with
            # error_measure.py and later filtered with filter.py
            with open(filename, "r") as file:
                c = 1
                for line in file.readlines():
                    energies.append(float(line.split(" ")[0]))
                    E_top_errors.append(float(line.split(" ")[1]))
                    E_bot_errors.append(float(line.split(" ")[2]))
                    ks.append(float(line.split(" ")[3]))
                    ks_top_errors.append(float(line.split(" ")[4]))
                    ks_bot_errors.append(float(line.split(" ")[5]))
                    kd.append(float(line.split(" ")[6]))
                    kd_top_errors.append(float(line.split(" ")[7]))
                    kd_bot_errors.append(float(line.split(" ")[8]))
                    pd.append(float(line.split(" ")[9]))
                    pd_top_errors.append(float(line.split(" ")[10]))
                    pd_bot_errors.append(float(line.split(" ")[11]))
                    test_number.append(c)
                    c += 1
                file.close()

            ################## FURTHER ERRORS COMPUTATION ##################
            ks_errors = [ks_bot_errors, ks_top_errors]
            kd_errors = [kd_bot_errors, kd_top_errors]
            pd_errors = [pd_bot_errors, pd_top_errors]

            if len(energies) > 0:

                mean_E = sum(energies) / len(energies)
                try:
                    stdev_E = statistics.stdev(energies) / math.sqrt(len(energies))
                except statistics.StatisticsError as error:
                    print(
                        "It seems that there is only 1 energy datapoint in file "
                        f"{filename}, but a minimum of 2 are needed to compute standard "
                        "deviations (errors). Exitting..."
                    )
                    sys.exit(0)
                mean_E_top_errors = sum(E_top_errors) / len(E_top_errors)
                mean_E_bot_errors = sum(E_bot_errors) / len(E_bot_errors)

                mean_ks = sum(ks) / len(ks)
                stdev_ks = statistics.stdev(ks) / math.sqrt(len(ks))
                mean_ks_top_errors = sum(ks_top_errors) / len(ks_top_errors)
                mean_ks_bot_errors = sum(ks_bot_errors) / len(ks_bot_errors)

                mean_kd = sum(kd) / len(kd)
                # Stochastic error (used for both upper and lower bounds)
                stdev_kd = statistics.stdev(kd) / math.sqrt(len(kd))
                # Oscillating error (upper bound)
                mean_kd_top_errors = sum(kd_top_errors) / len(kd_top_errors)
                # Oscillating error (lower bound)
                mean_kd_bot_errors = sum(kd_bot_errors) / len(kd_bot_errors)

                # This saves all the parameters necessary to compute the final
                # graph to the file 'joint_graph.txt'
                with open(joint_graph_file, "a") as file:
                    # E, E+, E-, Ks, Ks+, Ks-, Pd, Pd+, Pd-
                    file.write(
                        str(mean_E)
                        + " "
                        + str(stdev_E)
                        + " "
                        + str(mean_E_top_errors)
                        + " "
                        + str(mean_E_bot_errors)
                        + " "
                        + str(mean_ks)
                        + " "
                        + str(stdev_ks)
                        + " "
                        + str(mean_ks_top_errors)
                        + " "
                        + str(mean_ks_bot_errors)
                        + " "
                        + str(mean_kd)
                        + " "
                        + str(stdev_kd)
                        + " "
                        + str(mean_kd_top_errors)
                        + " "
                        + str(mean_kd_bot_errors)
                        + " \n"
                    )
                    file.close()
            else:
                print(f"File {file} has no data.")

        ################## FURTHER ERRORS DATA FETCHING ##################
        """We read the data of joint_graph_file, and compute the necessary 
        stuff for the joint plot."""
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
        with open(joint_graph_file, "r") as file:
            for line in file.readlines():
                energies_joint.append(float(line.split(" ")[0]))
                mean_E_errors.append(float(line.split(" ")[1]))
                E_top_errors_joint.append(float(line.split(" ")[2]))
                E_bot_errors_joint.append(float(line.split(" ")[3]))
                ks_joint.append(float(line.split(" ")[4]))
                mean_ks_errors.append(float(line.split(" ")[5]))
                ks_top_errors_joint.append(float(line.split(" ")[6]))
                ks_bot_errors_joint.append(float(line.split(" ")[7]))
                kd_joint.append(float(line.split(" ")[8]))
                mean_kd_errors.append(float(line.split(" ")[9]))
                kd_top_errors_joint.append(float(line.split(" ")[10]))
                kd_bot_errors_joint.append(float(line.split(" ")[11]))
            file.close()

        # We generate lists with the complete error data.
        E_errors_joint = [E_bot_errors_joint, E_top_errors_joint]
        ks_errors_joint = [ks_bot_errors_joint, ks_top_errors_joint]
        kd_errors_joint = [kd_bot_errors_joint, kd_top_errors_joint]
        E_osc_shade_top = [
            energies_joint[i] + abs(E_top_errors_joint[i])
            for i in range(len(energies_joint))
        ]
        E_osc_shade_bot = [
            energies_joint[i] - abs(E_bot_errors_joint[i])
            for i in range(len(energies_joint))
        ]
        E_stoch_shade_top = [
            energies_joint[i] + abs(mean_E_errors[i])
            for i in range(len(energies_joint))
        ]
        E_stoch_shade_bot = [
            energies_joint[i] - abs(mean_E_errors[i])
            for i in range(len(energies_joint))
        ]

        ks_osc_shade_top = [
            ks_joint[i] + abs(ks_top_errors_joint[i]) for i in range(len(ks_joint))
        ]
        ks_osc_shade_bot = [
            ks_joint[i] - abs(ks_bot_errors_joint[i]) for i in range(len(ks_joint))
        ]
        ks_stoch_shade_top = [
            ks_joint[i] + abs(mean_ks_errors[i]) for i in range(len(ks_joint))
        ]
        ks_stoch_shade_bot = [
            ks_joint[i] - abs(mean_ks_errors[i]) for i in range(len(ks_joint))
        ]

        kd_osc_shade_top = [
            kd_joint[i] + abs(kd_top_errors_joint[i]) for i in range(len(kd_joint))
        ]
        kd_osc_shade_bot = [
            kd_joint[i] - abs(kd_bot_errors_joint[i]) for i in range(len(kd_joint))
        ]
        kd_stoch_shade_top = [
            kd_joint[i] + abs(mean_kd_errors[i]) for i in range(len(kd_joint))
        ]
        kd_stoch_shade_bot = [
            kd_joint[i] - abs(mean_kd_errors[i]) for i in range(len(kd_joint))
        ]

        # We save the data in a global file.
        with open(f"plotting_data/master_plot_data_{arch}.txt", "a") as global_file:
            for (
                E,
                E_osc_bot,
                E_osc_top,
                E_stoch_bot,
                E_stoch_top,
                Fs,
                Fs_osc_bot,
                Fs_osc_top,
                Fs_stoch_bot,
                Fs_stoch_top,
                Fd,
                Fd_osc_bot,
                Fd_osc_top,
                Fd_stoch_bot,
                Fd_stoch_top,
            ) in zip(
                energies_joint,
                E_osc_shade_bot,
                E_osc_shade_top,
                E_stoch_shade_bot,
                E_stoch_shade_top,
                ks_joint,
                ks_osc_shade_bot,
                ks_osc_shade_top,
                ks_stoch_shade_bot,
                ks_stoch_shade_top,
                kd_joint,
                kd_osc_shade_bot,
                kd_osc_shade_top,
                kd_stoch_shade_bot,
                kd_stoch_shade_top,
            ):
                global_file.write(
                    f"{E} {E_osc_bot} {E_osc_top} {E_stoch_bot}"
                    f" {E_stoch_top} {Fs} {Fs_osc_bot} "
                    f"{Fs_osc_top} {Fs_stoch_bot} {Fs_stoch_top} "
                    f"{Fd} {Fd_osc_bot} {Fd_osc_top} "
                    f"{Fd_stoch_bot} {Fd_stoch_top}\n"
                )
            global_file.close()

        print(
            f"Error data of model {arch} nhid{Nhid} appended to "
            f"master_plot_data_{arch}.txt."
        )

############### DATA PREPARATION FOR PLOTTING ###############
if True:
    print("\n")
    data_list = [
        "E",
        "E_osc_bot",
        "E_osc_top",
        "E_stoch_bot",
        "E_stoch_top",
        "Fs",
        "Fs_osc_bot",
        "Fs_osc_top",
        "Fs_stoch_bot",
        "Fs_stoch_top",
        "Fd",
        "Fd_osc_bot",
        "Fd_osc_top",
        "Fd_stoch_bot",
        "Fd_stoch_top",
    ]

    dict_1sc = {}
    dict_2sc = {}
    dict_1sd = {}
    dict_2sd = {}
    dict_E = {}
    dict_F = {}
    dict_dict = {"1sc": dict_1sc, "2sc": dict_2sc, "1sd": dict_1sd, "2sd": dict_2sd}

    # We create the keys
    for e in data_list:
        key_1sc = f"1sc_{e}"
        key_2sc = f"2sc_{e}"
        key_1sd = f"1sd_{e}"
        key_2sd = f"2sd_{e}"
        dict_1sc[key_1sc] = []
        dict_2sc[key_2sc] = []
        dict_1sd[key_1sd] = []
        dict_2sd[key_2sd] = []

    for arch in net_archs:
        print(f"Loading dictionary of model {arch}...")
        with open(f"plotting_data/master_plot_data_{arch}.txt", "r") as f:
            list_of_keys = list(dict_dict[arch].keys())
            for line in f.readlines():
                line = line.split(" ")
                c = 0
                for item in line:
                    key_at_index = list_of_keys[c]
                    dict_dict[arch][key_at_index].append(float(item))
                    c += 1
            f.close()

        # We fill a dictionary with the energies and another one with the
        # fidelities
        for e in list(dict_dict[arch].keys()):
            if e.split("_")[1] == "E":
                dict_E[e] = dict_dict[arch][e]
            else:
                dict_F[e] = dict_dict[arch][e]

################################# PLOTTING #################################
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 5), sharey="row", sharex="col")
fig.tight_layout()
plt.subplots_adjust(wspace=0.12, hspace=0.1)
ax_E = axes[0]
ax_F = axes[1]
letters_E = ["(a)", "(b)", "(c)", "(d)"]
letters_F = ["(e)", "(f)", "(g)", "(h)"]

for arch in net_archs:
    index = net_archs.index(arch)
    x_axis = hidden_neurons if arch != "2sd" else [20, 32, 40, 60, 80, 100]

    # Energy
    axE = ax_E[index]
    if index == 0:
        axE.set_ylabel("E (Mev)", fontsize=15)
        axE.set_ylim(-2.227, -2.220)

    axE.set_title(f"{arch[0]} {arch[1:]}", fontsize=17)
    axE.set_xlim(20, 100)
    axE.tick_params(axis="y", labelsize=14)
    axE.tick_params(axis="both", which="both", direction="in")
    axE.tick_params(axis="both", which="major", width=1.7, length=4.5)
    axE.yaxis.set_minor_locator(MultipleLocator(0.001))

    axE.plot(x_axis, dict_E[f"{arch}_E"], label="Energy" if index == 0 else "")
    axE.axhline(
        y=-2.2267, color="red", linestyle="--", label="Exact" if index == 0 else ""
    )
    axE.annotate(
        text=letters_E[index], xy=(0.85, 0.85), xycoords="axes fraction", fontsize=17
    )
    if show_osc_error:
        axE.fill_between(
            x_axis,
            dict_E[f"{arch}_E_osc_bot"],
            dict_E[f"{arch}_E_osc_top"],
            edgecolor="#089FFF",
            facecolor="#089FFF",
            alpha=0.3,
        )
    if show_stoch_error:
        axE.fill_between(
            x_axis,
            dict_E[f"{arch}_E_stoch_bot"],
            dict_E[f"{arch}_E_stoch_top"],
            edgecolor="#089FFF",
            facecolor="#089FFF",
            alpha=0.5,
        )

    # Overlap
    axF = ax_F[index]
    if index == 0:
        d_shades = {"stoch": "#249225", "osc": "#abd5ab", "alpha": 0.5}
        axF.set_ylabel("Fidelity", fontsize=15)
        axF.set_ylim(0.9998, 1.000005)
        axF.yaxis.set_major_formatter(StrMethodFormatter("{x:,.5f}"))
        axF.yaxis.set_minor_locator(MultipleLocator(0.00001))

    axF.set_xlabel("$N_{\mathrm{hid}}$", fontsize=15)
    axF.set_xlim(20, 100)
    axF.tick_params(axis="both", labelsize=14)
    axF.tick_params(axis="both", which="both", direction="in")
    axF.tick_params(axis="both", which="major", width=1.7, length=4.5)
    axF.axhline(y=1.0, color="red", linestyle="--")
    axF.annotate(
        text=letters_F[index], xy=(0.85, 0.12), xycoords="axes fraction", fontsize=17
    )
    if show_osc_error:
        axF.fill_between(
            x_axis,
            dict_F[f"{arch}_Fs_osc_bot"],
            dict_F[f"{arch}_Fs_osc_top"],
            facecolor="purple",
            alpha=0.3,
        )
        axF.fill_between(
            x_axis,
            dict_F[f"{arch}_Fd_osc_bot"],
            dict_F[f"{arch}_Fd_osc_top"],
            facecolor=d_shades["osc"],
            alpha=d_shades["alpha"],
        )
    if show_stoch_error:
        axF.fill_between(
            x_axis,
            dict_F[f"{arch}_Fs_stoch_bot"],
            dict_F[f"{arch}_Fs_stoch_top"],
            facecolor="purple",
            alpha=0.5,
        )
        axF.fill_between(
            x_axis,
            dict_F[f"{arch}_Fd_stoch_bot"],
            dict_F[f"{arch}_Fd_stoch_top"],
            facecolor=d_shades["stoch"],
            alpha=d_shades["alpha"],
        )
    axF.plot(
        x_axis,
        dict_F[f"{arch}_Fs"],
        color="purple",
        label="S-state" if index == 0 else "",
    )
    axF.plot(
        x_axis,
        dict_F[f"{arch}_Fd"],
        color="green",
        label="D-state" if index == 0 else "",
    )

fig.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=5,
    fancybox=True,
    shadow=True,
    fontsize=14,
)

if save_plot:
    plt.savefig(path_of_plot, format="pdf", bbox_inches="tight")
    print(f"Figure saved in {path_of_plot}")

plt.show()
