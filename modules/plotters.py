#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:45:32 2022

@author: jozalen
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def pretraining_plots(x_axis,psi_s_pred,psi_d_pred,n_test,n_samples,psi_ansatz_s_normalized,psi_ansatz_d_normalized,
                      overlap_s,overlap_d,path_plot,t,s,exec_time=0.):
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(wspace=0.25,hspace=0.35)
    
    # L=0
    plt.subplot(2,2,1)
    plt.title("S-state wave functions. Epoch {}".format(t+1),fontsize=15)
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$",fontsize=15)
    plt.ylabel("$\psi\,(L=0)$",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_axis[:57],psi_s_pred.detach()[:57],label='$\psi_{\mathrm{ANN}}$')
    plt.plot(x_axis[:57],psi_ansatz_s_normalized.tolist()[:57],label='$\psi_{\mathrm{targ}}$')
    plt.legend(fontsize=15)
    
    # Overlap (L=0)
    plt.subplot(2,2,2)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_s[-1]),fontsize=15)
    plt.xlabel("Epoch",fontsize=15)
    plt.ylabel("Overlap$\,(L=0)$",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.linspace(0,len(overlap_s),len(overlap_s)),overlap_s,label='$F^{S}$')
    plt.legend(fontsize=15)
    
    # L=2
    plt.subplot(2,2,3)
    plt.title("D-state wave functions. Epoch {}".format(t+1),fontsize=15)
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$",fontsize=15)
    plt.ylabel("$\psi\,(L=2)$",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(x_axis[:57],psi_d_pred.detach()[:57],label='$\psi_{\mathrm{ANN}}$')
    plt.plot(x_axis[:57],psi_ansatz_d_normalized.tolist()[:57],label='$\psi_{\mathrm{targ}}$')
    plt.legend(fontsize=15)
    
    # Overlap (L=2)
    plt.subplot(2,2,4)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_d[-1]),fontsize=15)
    plt.xlabel("Epoch",fontsize=15)
    plt.ylabel("Overlap$\,(L=2)$",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.linspace(0,len(overlap_d),len(overlap_d)),overlap_d,label='$F^{D}$')
    plt.legend(fontsize=15)

    if s == True:
        plot_path = path_plot + 'time{}.png'.format(exec_time)
        plt.savefig(plot_path)
        #print('\nPlot saved in {}'.format(plot_path))
    plt.pause(0.001)

def minimisation_plots(x_axis,ann_s,ann_d,psi_exact_s_normalized,psi_exact_d_normalized,ks_accum,kd_accum,ks,kd,
                       K_accum,K,U,E,U_accum,E_accum,E_exact,pd,pd_accum,PD_exact,path_plot,t,s,exec_time=0.):
    plt.figure(figsize=(18,11))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    k = x_axis
    
    # L=0
    plt.subplot(2,4,1)
    plt.title("S-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=0)$")
    plt.plot(k[:53],ann_s.detach()[:53],label='$\psi_{ANN}^{L=0}$')
    plt.plot(k[:53],psi_exact_s_normalized.numpy()[:53],label='$\psi_{targ}^{L=0}$')
    plt.legend(fontsize=17)
    
    # L=2
    plt.subplot(2,4,2)
    plt.title("D-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=2)$")
    plt.plot(k[:53],ann_d.detach()[:53],label='$\psi_{ANN}^{L=2}$')
    plt.plot(k[:53],psi_exact_d_normalized.numpy()[:53],label='$\psi_{targ}^{L=2}$')
    plt.legend(fontsize=17)
    
    # Overlap
    plt.subplot(2,4,3)
    plt.title("Overlap")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.ylim(0.9995,1.0)
    plt.plot(torch.linspace(0,len(ks_accum),len(ks_accum)).numpy(),ks_accum,label='$K^S={:4.6f}$'.format(ks))
    plt.plot(torch.linspace(0,len(kd_accum),len(kd_accum)).numpy(),kd_accum,label='$K^D={:4.6f}$'.format(kd))
    plt.legend(fontsize=12)
    
    # K
    plt.subplot(2,4,4)
    plt.title("Kinetic Energy")
    plt.xlabel("Epoch")
    plt.ylabel("K (MeV)")
    plt.ylim(13,17)
    plt.plot(torch.linspace(0,len(K_accum),len(K_accum)).numpy(),K_accum,label='$K={:3.6f}\,(MeV)$'.format(K))
    plt.legend(fontsize=12)
    
    # U
    plt.subplot(2,4,5)
    plt.title("Potential Energy")
    plt.xlabel("Epoch")
    plt.ylabel("U (MeV)")
    plt.ylim(-18,-15)
    plt.plot(torch.linspace(0,len(U_accum),len(U_accum)).numpy(),U_accum,label='$U={:3.6f}\,(MeV)$'.format(U))
    plt.legend(fontsize=12)
    
    # E
    plt.subplot(2,4,6)
    plt.title("Total Energy")
    plt.xlabel("Epoch")
    plt.ylabel("E (MeV)")
    plt.ylim(-2.227,-2.220)
    plt.plot(torch.linspace(0,len(E_accum),len(E_accum)).numpy(),E_accum,label='$E={:3.6f}\,(MeV)$'.format(E))
    plt.axhline(y=E_exact, color="red", linestyle="--", label="Exact")
    plt.legend(fontsize=12)
    
    # PD
    plt.subplot(2,4,7)
    plt.title("Probability of D-state")
    plt.xlabel("Epoch")
    plt.ylabel("P (%)")
    plt.ylim(4,5)
    plt.plot(torch.linspace(0,len(pd_accum),len(pd_accum)).numpy(),pd_accum,label='$P={:3.4f}$ %'.format(pd))
    plt.axhline(y=PD_exact, color="red", linestyle="--", label="Exact")
    plt.legend(fontsize=12)

    if s == True:
        full_path_plot = '{}time{}.png'.format(path_plot,exec_time)
        plt.savefig(full_path_plot)
        #print('\nPlot saved in {}'.format(path_plot))
    plt.pause(0.001)
    
def error_measure_plots(adaptive_lims,ks_accum,factor_k_sup,factor_k_inf,kd_accum,ks,kd,mean_ks,mean_ks_top,mean_ks_bot,
                mean_kd,mean_kd_bot,mean_kd_top,factor_E_sup,factor_E_inf,E_accum,E,E_exact,mean_E,mean_E_top,
                mean_E_bot,factor_pd_sup,factor_pd_inf,pd_accum,pd,PD_exact,mean_PD,mean_pd_top,mean_pd_bot,
                periodic_plots):
    
    plt.figure(figsize=(18,8))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
        
    # Overlap
    plt.subplot(1,3,1)
    plt.title("Overlap")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    if adaptive_lims == False:
        plt.ylim(0.9998,1.0)
    elif adaptive_lims == True:
        cand_to_lim_sup_k = factor_k_sup*max(ks_accum[20:])
        if cand_to_lim_sup_k >1. :
            lim_sup_k = 1.
        else:
            lim_sup_k = cand_to_lim_sup_k
        lim_inf_k = factor_k_inf*min(kd_accum[20:])
        plt.ylim(lim_inf_k,lim_sup_k)
    plt.plot(torch.linspace(0,len(ks_accum),len(ks_accum)).numpy(),ks_accum,label='$K^S={:4.6f}$'.format(ks))
    plt.plot(torch.linspace(0,len(kd_accum),len(kd_accum)).numpy(),kd_accum,label='$K^D={:4.6f}$'.format(kd))
        # Mean ks
    plt.axhline(y=mean_ks, color="green", linestyle="--", label='Mean $K^S={:15.8f}$'.format(mean_ks))
    #plt.plot(torch.linspace(20,len(ks_accum[21:]),len(ks_accum[21:])).numpy(),mean_ks_accum,label='Mean ks')
            # Max value
    plt.axhline(y=mean_ks_top, color="orange", linestyle="--", label='$K^S_t={:15.8f}$'.format(mean_ks_top))
            # Min value
    plt.axhline(y=mean_ks_bot, color="purple", linestyle="--", label='$K^S_b={:15.8f}$'.format(mean_ks_bot))
        # Mean kd
    plt.axhline(y=mean_kd, color="green", linestyle="--", label='Mean $K^D={:15.8f}$'.format(mean_kd))
    #plt.plot(torch.linspace(20,len(kd_accum[21:]),len(kd_accum[21:])).numpy(),mean_kd_accum,label='Mean kd')
            # Max value
    plt.axhline(y=mean_kd_top, color="orange", linestyle="--", label='$K^S_t={:15.8f}$'.format(mean_kd_top))
            # Min value
    plt.axhline(y=mean_kd_bot,  color="purple", linestyle="--", label='$K^D_b={:15.8f}$'.format(mean_kd_bot))
           
    # E
    plt.subplot(1,3,2)
    plt.title("Total Energy")
    plt.xlabel("Epoch")
    plt.ylabel("E (MeV)")
    if adaptive_lims == False:
        plt.ylim(-2.221,-2.2) # fixed ylim
    elif adaptive_lims == True:
        lim_sup_E = factor_E_sup*max(E_accum[20:])
        lim_inf_E = factor_E_inf*min(E_accum[20:])
        plt.ylim(lim_inf_E,lim_sup_E)
    plt.plot(torch.linspace(0,len(E_accum),len(E_accum)).numpy(),E_accum,label='$E={:3.6f}$'.format(E))
    plt.axhline(y=E_exact, color="red", linestyle="--", label="Exact")
        # Mean
    plt.axhline(y=mean_E, color="green", linestyle="--", label='Mean=${:15.6f}$'.format(mean_E))
    #plt.plot(torch.linspace(20,len(E_accum[21:]),len(E_accum[21:])).numpy(),mean_E_accum,label='Mean')
            # Max value
    plt.axhline(y=mean_E_top, color="orange", linestyle="--", label='$E_t={:15.6f}$'.format(mean_E_top))
            # Min value
    plt.axhline(y=mean_E_bot,  color="purple", linestyle="--", label='$E_b={:15.6f}$'.format(mean_E_bot))
    
    plt.legend(fontsize=12)
    
    # PD
    plt.subplot(1,3,3)
    plt.title("Probability of D-state")
    plt.xlabel("Epoch")
    plt.ylabel("P (%)")
    if adaptive_lims == False:
        plt.ylim(4.3,4.7)
    elif adaptive_lims == True:
        lim_sup_pd = factor_pd_sup*max(pd_accum[20:])
        lim_inf_pd = factor_pd_inf*min(pd_accum[20:])
        plt.ylim(lim_inf_pd,lim_sup_pd)
    plt.plot(torch.linspace(0,len(pd_accum),len(pd_accum)).numpy(),pd_accum,label='$P={:3.4f}$'.format(pd))
    plt.axhline(y=PD_exact, color="red", linestyle="--", label="Exact")
        # Error
    plt.axhline(y=mean_PD, color="green", linestyle="--", label='Mean=${:15.4f}$'.format(mean_PD))
    #plt.plot(torch.linspace(20,len(pd_accum[21:]),len(pd_accum[21:])).numpy(),mean_PD_accum,label='Mean')
            # Max value
    plt.axhline(y=mean_pd_top, color="orange", linestyle="--", label='$P_t={:15.4f}$'.format(mean_pd_top))
            # Min value
    plt.axhline(y=mean_pd_bot, color="purple", linestyle="--", label='$P_b={:15.4f}$'.format(mean_pd_bot))
    
    
    plt.legend(fontsize=12)
    
    plt.suptitle("Mean value and error analysis")
    plt.pause(0.001)