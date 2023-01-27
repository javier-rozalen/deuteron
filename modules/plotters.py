# -*- coding: utf-8 -*-
######################## IMPORTS ########################
import torch, gc
import matplotlib.pyplot as plt
import numpy as np

######################## MISC. ########################
plt.rcParams['agg.path.chunksize'] = 10000

######################## PLOTTER FUNCTIONS ########################
def pretraining_plots(x_axis, psi_s_pred, psi_d_pred, n_samples,
                      psi_ansatz_s_normalized, psi_ansatz_d_normalized,
                      overlap_s, overlap_d, path_plot, t, s, exec_time=0.,
                      show=True):
    """
    Plots the wave function and overlap evolution of the pretraining phase, 
    for both the S and D states.

    Parameters
    ----------
    x_axis : tensor, float
        X axis.
    psi_s_pred : tensor, float
        ANN prediction for the S-state.
    psi_d_pred : tensor, float
        ANN prediction for the D-state.
    n_samples : int
        Number of samples. Do not touch!
    psi_ansatz_s_normalized : tensor, float
        Normalized S-state wave function ansatz.
    psi_ansatz_d_normalized : tensor, float
        Normalized S-state wave function ansatz.
    overlap_s : tensor, float
        Overlap between the ANN S-state and the ansatz S-state.
    overlap_d : tensor, float
        Overlap between the ANN D-state and the ansatz D-state.
    path_plot : string
        Path where the plot will be stored, if pertinent. 
    t : int
        Current epoch.
    s : Boolean
        Wether the plot is saved or not.
    exec_time : float, optional
        Execution time. The default is 0..
    show : Boolean, optional
        Wether the plot is shown as stdout or not. The default is True.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    
    # L=0
    plt.subplot(2, 2, 1)
    plt.title("S-state wave functions. Epoch {}".format(t+1), fontsize=15)
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$", fontsize=15)
    plt.ylabel("$\psi\,(L=0)$", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_axis[:57], psi_s_pred.detach()[:57], 
             label='$\psi_{\mathrm{ANN}}$')
    plt.plot(x_axis[:57], psi_ansatz_s_normalized.tolist()[:57], 
             label='$\psi_{\mathrm{targ}}$')
    plt.legend(fontsize=15)
    
    # Overlap (L=0)
    plt.subplot(2, 2, 2)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_s[-1]), 
              fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Overlap$\,(L=0)$", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.linspace(0, len(overlap_s), len(overlap_s)), overlap_s, 
             label='$F^{S}$')
    plt.legend(fontsize=15)
    
    # L=2
    plt.subplot(2, 2, 3)
    plt.title("D-state wave functions. Epoch {}".format(t+1), fontsize=15)
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$", fontsize=15)
    plt.ylabel("$\psi\,(L=2)$", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_axis[:57], psi_d_pred.detach()[:57], 
             label='$\psi_{\mathrm{ANN}}$')
    plt.plot(x_axis[:57], psi_ansatz_d_normalized.tolist()[:57], 
             label='$\psi_{\mathrm{targ}}$')
    plt.legend(fontsize=15)
    
    # Overlap (L=2)
    plt.subplot(2, 2, 4)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_d[-1]), 
              fontsize=15)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Overlap$\,(L=2)$", fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.linspace(0, len(overlap_d), len(overlap_d)), overlap_d, 
             label='$F^{D}$')
    plt.legend(fontsize=15)

    if s == True:
        #plot_path = path_plot + 'time{}.png'.format(exec_time)
        plot_path = path_plot + '.pdf'
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        #print('\nPlot saved in {}'.format(plot_path))
    if show:
        plt.pause(0.001)

    plt.close()
    
def minimisation_plots(x_axis, ann_s, ann_d, psi_exact_s_normalized,
                       psi_exact_d_normalized, ks_accum, kd_accum, ks, kd,
                       K_accum, K, U, E, U_accum, E_accum, E_exact, pd, 
                       pd_accum, PD_exact, path_plot, t, s, show):
    """
    Plots the evolution of the energy (kinetic, potential and total), 
    wave function, overlap with the exact wave function for both the S and D
    states. The probability of the D-state is also plotted.

    Parameters
    ----------
    x_axis : tensor
        X axis.
    ann_s : tensor
        ANN prediction for the S-state.
    ann_d : tensor
        ANN prediction for the S-state.
    psi_exact_s_normalized : tensor
        Exact S-state wave function (obtained via exact diagonalization).
    psi_exact_d_normalized : tensor
        Exact D-state wave function (obtained via exact diagonalization).
    ks_accum : tensor
        Overlap of the S-state calculated until the current epoch.
    kd_accum : tensor
        Overlap of the D-state calculated until the current epoch.
    ks : tensor
        Overlap of the S-state.
    kd : tensor
        Overlap of the D-state.
    K_accum : tensor
        Kinetic energy calculated until the current epoch.
    K : tensor
        Kinetic energy at the current epoch.
    U : tensor
        Potential energy at the current epoch.
    E : tensor
        Total energy at the current epoch.
    U_accum : tensor
        Potential energy calculated until the current epoch.
    E_accum : tensor
        Total energy calculated until the current epoch.
    E_exact : tensor
        Exact total energy (obtained via exact diagonalization).
    pd : tensor
        Probability of the D-state at the current epoch.
    pd_accum : tensor
        Probability of the D-state calculated until the current epoch.
    PD_exact : tensor
        Exact probability of the D-state (obtained via exact diagonalization).
    path_plot : string
        Path where the plot will be stored, if pertinent.
    t : int
        Current epoch number.
    s : Boolean
        Wether the plot is saved or not.
    show : Boolean
        Wether the plot is shown as stdout or not. 

    Returns
    -------
    None.

    """
    # matplotlib.use('Agg')
    plt.figure(figsize=(18, 11))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    k = x_axis
    
    # L=0
    plt.subplot(2, 4, 1)
    plt.title("S-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=0)$")
    plt.plot(k[:53], ann_s.detach()[:53], label='$\psi_{ANN}^{L=0}$')
    plt.plot(k[:53], psi_exact_s_normalized.numpy()[:53], 
             label='$\psi_{targ}^{L=0}$')
    plt.legend(fontsize=17)
    
    # L=2
    plt.subplot(2, 4, 2)
    plt.title("D-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=2)$")
    plt.plot(k[:53], ann_d.detach()[:53], label='$\psi_{ANN}^{L=2}$')
    plt.plot(k[:53], psi_exact_d_normalized.numpy()[:53], 
             label='$\psi_{targ}^{L=2}$')
    plt.legend(fontsize=17)
    
    # Overlap
    plt.subplot(2, 4, 3)
    plt.title("Overlap")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.ylim(0.9995, 1.0)
    plt.plot(torch.linspace(0, len(ks_accum), len(ks_accum)).numpy(), ks_accum, 
             label='$K^S={:4.6f}$'.format(ks))
    plt.plot(torch.linspace(0, len(kd_accum), len(kd_accum)).numpy(), kd_accum, 
             label='$K^D={:4.6f}$'.format(kd))
    plt.legend(fontsize=12)
    
    # K
    plt.subplot(2, 4, 4)
    plt.title("Kinetic Energy")
    plt.xlabel("Epoch")
    plt.ylabel("K (MeV)")
    plt.ylim(13, 17)
    plt.plot(torch.linspace(0, len(K_accum), len(K_accum)).numpy(), K_accum, 
             label='$K={:3.6f}\,(MeV)$'.format(K))
    plt.legend(fontsize=12)
    
    # U
    plt.subplot(2, 4, 5)
    plt.title("Potential Energy")
    plt.xlabel("Epoch")
    plt.ylabel("U (MeV)")
    plt.ylim(-18, -15)
    plt.plot(torch.linspace(0, len(U_accum), len(U_accum)).numpy(), U_accum, 
             label='$U={:3.6f}\,(MeV)$'.format(U))
    plt.legend(fontsize=12)
    
    # E
    plt.subplot(2, 4, 6)
    plt.title("Total Energy")
    plt.xlabel("Epoch")
    plt.ylabel("E (MeV)")
    plt.ylim(-2.227, -2.220)
    plt.plot(torch.linspace(0, len(E_accum), len(E_accum)).numpy(), E_accum, 
             label='$E={:3.6f}\,(MeV)$'.format(E))
    plt.axhline(y=E_exact, color="red", linestyle="--", label="Exact")
    plt.legend(fontsize=12)
    
    # PD
    plt.subplot(2, 4, 7)
    plt.title("Probability of D-state")
    plt.xlabel("Epoch")
    plt.ylabel("P (%)")
    plt.ylim(4, 5)
    plt.plot(torch.linspace(0, len(pd_accum), len(pd_accum)).numpy(), pd_accum, 
             label='$P={:3.4f}$ %'.format(pd))
    plt.axhline(y=PD_exact, color="red", linestyle="--", label="Exact")
    plt.legend(fontsize=12)
    
    if s == True:
        full_path_plot = f'{path_plot}.pdf'
        plt.savefig(full_path_plot, format='pdf', bbox_inches='tight')
        
    if show == True:
        plt.pause(0.001)
    
    plt.clf()
    plt.close('all')
    gc.collect()
    
def error_measure_plots(adaptive_lims, ks_accum, factor_k_sup, factor_k_inf, 
                        kd_accum, ks, kd, mean_ks, mean_ks_top, mean_ks_bot, 
                        mean_kd, mean_kd_bot, mean_kd_top, factor_E_sup, 
                        factor_E_inf, E_accum, E, E_exact, mean_E, mean_E_top, 
                        mean_E_bot, factor_pd_sup, factor_pd_inf, pd_accum, pd, 
                        PD_exact, mean_PD, mean_pd_top, mean_pd_bot, show=True):
    """
    Plots the error bars of the different wave energies and overlaps, thereby
    depicting our way of calculating them.

    Parameters
    ----------
    adaptive_lims : Boolean
        Wether the plot limits are adaptive or fixed.
    ks_accum : tensor
        Overlap of the S-state calculated until the current epoch.
    factor_k_sup : float
        Multiplicative factor implicated in the adaptive limits.
    factor_k_inf : float
        Multiplicative factor implicated in the adaptive limits.
    kd_accum : tensor
        Overlap of the D-state calculated until the current epoch.
    ks : tensor
        Overlap of the S-state at the current epoch.
    kd : tensor
        Overlap of the D-state at the current epoch.
    mean_ks : tensor
        Arithmetic mean of all the ks (taken from the different seeds).
    mean_ks_top : tensor
        Arithmetic mean of all the top-error ks.
    mean_ks_bot : tensor
        Arithmetic mean of all the bottom-error ks.
    mean_kd : tensor
        Arithmetic mean of all the kd (taken from the different seeds).
    mean_kd_top : tensor
        Arithmetic mean of all the top-error kd.
    mean_kd_bot : tensor
        Arithmetic mean of all the bottom-error kd.
    factor_E_sup : tensor
        Multiplicative factor implicated in the adaptive limits.
    factor_E_inf : tensor
        Multiplicative factor implicated in the adaptive limits.
    E_accum : tensor
        Total energy calculated until the current epoch.
    E : tensor
        Total energy calculated at the current epoch.
    E_exact : tensor
        Exact total energy (obtained via exact diagonalization).
    mean_E : tensor
        Arithmetic mean of all the Es (taken from the different seeds).
    mean_E_top : tensor
        Arithmetic mean of all the top-error Es.
    mean_E_bot : tensor
        Arithmetic mean of all the bottom-error Es.
    factor_pd_sup : tensor
        Multiplicative factor implicated in the adaptive limits.
    factor_pd_inf : tensor
        Multiplicative factor implicated in the adaptive limits.
    pd_accum : tensor
        Probability of the D-state calculated until the current epoch.
    pd : tensor
        Probability of the D-state at the current epoch.
    PD_exact : tensor
        Exact probability of the D-state.
    mean_PD : tensor
        Arithmetic mean of all the pds (taken from the different seeds).
    mean_pd_top : tensor
        Arithmetic mean of all the top-error pds.
    mean_pd_bot : tensor
        Arithmetic mean of all the bottom-error pds.
    periodic_plots : Boolean
        Wether periodic plots are shown or not.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(18, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
    # Overlap
    plt.subplot(1, 3, 1)
    plt.title("Overlap")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    if adaptive_lims == False:
        plt.ylim(0.9998, 1.0)
    elif adaptive_lims == True:
        cand_to_lim_sup_k = factor_k_sup*max(ks_accum[20:])
        if cand_to_lim_sup_k > 1. :
            lim_sup_k = 1.
        else:
            lim_sup_k = cand_to_lim_sup_k
        lim_inf_k = factor_k_inf*min(kd_accum[20:])
        plt.ylim(lim_inf_k, lim_sup_k)
    plt.plot(torch.linspace(0, len(ks_accum), len(ks_accum)).numpy(), ks_accum,
             label='$K^S={:4.6f}$'.format(ks))
    plt.plot(torch.linspace(0, len(kd_accum), len(kd_accum)).numpy(), kd_accum,
             label='$K^D={:4.6f}$'.format(kd))
        # Mean ks
    plt.axhline(y=mean_ks, color="green", linestyle="--", 
                label='Mean $K^S={:15.8f}$'.format(mean_ks))
    #plt.plot(torch.linspace(20,len(ks_accum[21:]),len(ks_accum[21:])).numpy(),mean_ks_accum,label='Mean ks')
            # Max value
    plt.axhline(y=mean_ks_top, color="orange", linestyle="--", 
                label='$K^S_t={:15.8f}$'.format(mean_ks_top))
            # Min value
    plt.axhline(y=mean_ks_bot, color="purple", linestyle="--", 
                label='$K^S_b={:15.8f}$'.format(mean_ks_bot))
        # Mean kd
    plt.axhline(y=mean_kd, color="green", linestyle="--", 
                label='Mean $K^D={:15.8f}$'.format(mean_kd))
    #plt.plot(torch.linspace(20,len(kd_accum[21:]),len(kd_accum[21:])).numpy(),mean_kd_accum,label='Mean kd')
            # Max value
    plt.axhline(y=mean_kd_top, color="orange", linestyle="--", 
                label='$K^S_t={:15.8f}$'.format(mean_kd_top))
            # Min value
    plt.axhline(y=mean_kd_bot,  color="purple", linestyle="--", 
                label='$K^D_b={:15.8f}$'.format(mean_kd_bot))
           
    # E
    plt.subplot(1, 3, 2)
    plt.title("Total Energy")
    plt.xlabel("Epoch")
    plt.ylabel("E (MeV)")
    if adaptive_lims == False:
        plt.ylim(-2.221, -2.2) # fixed ylim
    elif adaptive_lims == True:
        lim_sup_E = factor_E_sup*max(E_accum[20:])
        lim_inf_E = factor_E_inf*min(E_accum[20:])
        plt.ylim(lim_inf_E, lim_sup_E)
    plt.plot(torch.linspace(0, len(E_accum), len(E_accum)).numpy(), E_accum,
             label='$E={:3.6f}$'.format(E))
    plt.axhline(y=E_exact, color="red", linestyle="--", label="Exact")
        # Mean
    plt.axhline(y=mean_E, color="green", linestyle="--", 
                label='Mean=${:15.6f}$'.format(mean_E))
    #plt.plot(torch.linspace(20,len(E_accum[21:]),len(E_accum[21:])).numpy(),mean_E_accum,label='Mean')
            # Max value
    plt.axhline(y=mean_E_top, color="orange", linestyle="--", 
                label='$E_t={:15.6f}$'.format(mean_E_top))
            # Min value
    plt.axhline(y=mean_E_bot,  color="purple", linestyle="--", 
                label='$E_b={:15.6f}$'.format(mean_E_bot))
    
    plt.legend(fontsize=12)
    
    # PD
    plt.subplot(1, 3, 3)
    plt.title("Probability of D-state")
    plt.xlabel("Epoch")
    plt.ylabel("P (%)")
    if adaptive_lims == False:
        plt.ylim(4.3, 4.7)
    elif adaptive_lims == True:
        lim_sup_pd = factor_pd_sup*max(pd_accum[20:])
        lim_inf_pd = factor_pd_inf*min(pd_accum[20:])
        plt.ylim(lim_inf_pd, lim_sup_pd)
    plt.plot(torch.linspace(0, len(pd_accum), len(pd_accum)).numpy(), pd_accum,
             label='$P={:3.4f}$'.format(pd))
    plt.axhline(y=PD_exact, color="red", linestyle="--", label="Exact")
        # Error
    plt.axhline(y=mean_PD, color="green", linestyle="--", 
                label='Mean=${:15.4f}$'.format(mean_PD))
    #plt.plot(torch.linspace(20, len(pd_accum[21:]), len(pd_accum[21:])).numpy(),
    #mean_PD_accum, label='Mean')
            # Max value
    plt.axhline(y=mean_pd_top, color="orange", linestyle="--", 
                label='$P_t={:15.4f}$'.format(mean_pd_top))
            # Min value
    plt.axhline(y=mean_pd_bot, color="purple", linestyle="--", 
                label='$P_b={:15.4f}$'.format(mean_pd_bot))
    
    plt.legend(fontsize=12)
    
    plt.suptitle("Mean value and error analysis")
    
    if show:
       plt.pause(0.001)
