# -*- coding: utf-8 -*-
######### IMPORTS ##########
import torch
from modules.physical_constants import hbar, mu

############################ LOSS FUNCTIONS ##############################
def overlap(model, arch, train_data, q_2, integration, w_i, norm_s, norm_d, 
         psi_ansatz_s, psi_ansatz_d):
    """
    Computes the cost, which is computed as the overlap between the neural
    network and the ansatz wave functions.

    Parameters
    ----------
    model : object
        Neural Network (torch) instance.
    arch : string
        Architecture among the 4 specified in the article.
    train_data : tensor
        Training data.
    q_2 : tensor
        Training data squared. We pass it as argument so as not to over-compute 
        it.
    integration : module
        Integration module in turn imported in the main script.
    w_i : tensor
        Integration weights.
    norm_s : tensor
        Norm of the ansatz S-state.
    norm_d : tensor
        Norm of the ansatz S-state.
    psi_ansatz_s : tensor
        Ansatz of the S-state. 
    psi_ansatz_d : tensor
        Ansatz of the D-state.

    Returns
    -------
    overlap: tensor
        Overlap between the NN ansätze and the analytical ansätze.
    psi_s_pred : tensor
        NN prediction for the S-state.
    psi_d_pred : tensor
        NN prediction for the D-state.
    k_s : tensor
        Overlap between the NN S-state ansatz and the analytical S-state 
        ansatz.
    k_d : tensor
        Overlap between the NN D-state ansatz and the analytical D-state 
        ansatz.

    """                   
    ann = model(train_data.clone().unsqueeze(1))
    if arch[2] == 'c':
        ann1, ann2 = ann[:,:1].squeeze(), ann[:,1:].squeeze()
    else:
        ann1, ann2 = ann[0], ann[1]
    
    # C (L=0)
    num = ann1*psi_ansatz_s*q_2
    denom1 = q_2*(ann1**2) 
    nume = (integration.gl64(w_i, num))**2
    den1 = integration.gl64(w_i, denom1)
    den2 = norm_s
    k_s = nume/(den1*den2)
    c_s = (k_s - torch.tensor(1.))**2
    norm_s_ann = den1
    
    # C (L=2)
    num = ann2*psi_ansatz_d*q_2
    denom1 = q_2*(ann2**2) 
    nume = (integration.gl64(w_i, num))**2
    den1 = integration.gl64(w_i, denom1)
    den2 = norm_d
    k_d = nume/(den1*den2)
    c_d = (k_d - torch.tensor(1.))**2
    norm_d_ann = den1
    
    psi_s_pred = ann1/torch.sqrt(norm_s_ann)
    psi_d_pred = ann2/torch.sqrt(norm_d_ann)
    
    overlap = c_s + c_d
    
    return overlap, psi_s_pred, psi_d_pred, k_s, k_d


def energy(model, train_data, q_2, arch, integration, w_i, q2_128, 
           wi_128, V_ij, psi_exact_s, psi_exact_s_norm, psi_exact_d, 
           psi_exact_d_norm):
    """
    Returns the cost computed as the expected energy using the current 
    wavefunction (ANN). Also returns the overlap with the theoretical 
    wavefunction.

    Parameters
    ----------
    model : object
        Neural Network (torch) instance.
    train_data : tensor
        Training data.
    q_2 : tensor
        Training data squared. We pass it as argument so as not to over-compute 
        it.
    arch : string
        Architecture among the 4 specified in the article.
    integration : module
        Integration module in turn imported in the main script.
    w_i : tensor
        Integration weights.
    q2_128 : tensor
        Same as q_2 but duplicated in length (copy-paste).
    wi_128 : tensor
        Same as q_2 but duplicated in length (copy-paste).
    V_ij : tensor
        Potential matrix elements.
    psi_exact_s : tensor
        Exact diagonalization S-state.
    psi_exact_s_norm : tensor
        Norm of psi_exact_s.
    psi_exact_d : tensor
        Exact diagonalization D-state.
    psi_exact_d_norm : tensor
        Norm of psi_exact_d.

    Returns
    -------
    E : tensor
        Total energy of the ANN prediction.
    ann_s : tensor
        ANN prediction for the S-state.
    ann_d : tensor
        ANN prediction for the D-state.
    norm2 : tensor
        Squared norm of the prediction wave function.
    K : tensor
        Kinetic energy of the ANN prediction.
    U : tensor
        Potential energy of the ANN prediction.
    ks : tensor
        Overlap between the ANN S-state and the exact S-state.
    kd : tensor
        Overlap between the ANN D-state and the exact D-state.
    pd : tensor
        Probability of the ANN D-state.

    """

    ann = model(train_data.clone().unsqueeze(1))
    if arch[2] == 'c':
        ann1, ann2 = ann[:,:1].squeeze(), ann[:,1:].squeeze()
    else:
        ann1, ann2 = ann[0], ann[1]

    # Norm
    norm_s = integration.gl64(q_2*(ann1)**2, w_i)
    norm_d = integration.gl64(q_2*(ann2)**2, w_i)    
    norm2 = norm_s + norm_d # squared norm
    
    # Wave function
    ann_s = ann1/torch.sqrt(norm2)
    ann_d = ann2/torch.sqrt(norm2)
    
    psi_128 = torch.cat((ann_s, ann_d)) # S-D concatenated wavefunction
    y = psi_128*q2_128*wi_128 # auxiliary tensor
    
    K = ((hbar**2)/mu)*torch.dot(wi_128, (psi_128*q2_128)**2) # Kinetic energy
    U = torch.matmul(y, torch.matmul(V_ij, y)) # U
    E = K + U # E
    ks = ((integration.gl64(q_2*ann1*psi_exact_s, w_i))**2)/ \
        (norm_s*psi_exact_s_norm) # Overlap (L=0)
    kd = ((integration.gl64(q_2*ann2*psi_exact_d, w_i))**2)/ \
        (norm_d*psi_exact_d_norm) # Overlap (L=2)
    pd = 100.*norm_d/norm2 # prob. of D-state
    
    return E, ann_s, ann_d, norm2, K, U, E, ks, kd, pd