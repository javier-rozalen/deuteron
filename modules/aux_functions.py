############# CONTENTS OF THE FILE #############
"""
This file is intended to be a library that can be imported at any time by the 
main scripts. It contains different functions which are used repeatedly over
the ANN training and post-processing. Each function has a documentation 
text. 
"""

############# IMPORTS #############
import re, os
from math import log10, floor

############# FUNCTIONS #############
def chunks(l, n):
    """
    Splits an input list l into chunks of size n.

    Parameters
    ----------
    l : list
        List to be splitted.
    n : int
        Size of each chunk.

    Returns
    -------
    splitted_list: list
        Splitted list.

    """
    n = max(1, n)   
    splitted_list = [l[i:i+n] for i in range(0, len(l), n)]
    return splitted_list

def split(l, n):
    """
    Splits an input list l into n chunks.
    
    Parameters
    ----------
    l : list
        List to be splitted.
    n : TYPE
        Number of chunks.

    Returns
    -------
    splitted_list: list
        Splitted list.

    """
    n = min(n, len(l)) # don't create empty buckets
    k, m = divmod(len(l), n)
    splitted = (l[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    splitted_list = list(splitted)
    return splitted_list

def num_sort(test_string):
    """
    Sorts a given list of strings numerically.

    Parameters
    ----------
    test_string : list
        List of strings.

    Returns
    -------
    sorted_list : list
        Numerically-sorted list.

    """
    sorted_list = list(map(int, re.findall(r'\d+', test_string)))[0]
    
    return sorted_list

def get_keys_from_value(d, val):
    """
    Given a dictionary and a value, it returns the corresponding key.

    Parameters
    ----------
    d : dict
        Input dictionary.
    val : any
        Dictionary value that we want to get the key of.

    Returns
    -------
    keys_list: list
        List of the keys that correspond to 'val'.

    """
    keys_list = [k for k, v in d.items() if v == val]
    return keys_list

def show_layers(model):
    """
    Shows the layers of the input Neural Network model.

    Parameters
    ----------
    model : torch object NN
        NN model.

    Returns
    -------
    None.

    """
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : " \
              f"{param[:100]} \n")
        
def dir_support(list_of_nested_dirs):
    """
    Directories support: ensures that the (nested) directories given via the 
    input list do exist, creating them if necessary. 

    Parameters
    ----------
    list_of_nested_dirs : list
        Nested directories in order.

    Returns
    -------
    None.

    """
    for i in range(len(list_of_nested_dirs)):
        potential_dir = '/'.join(list_of_nested_dirs[:i+1]) 
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)
            print(f'Creating directory {potential_dir}...')
            
def round_to_1(x):
    """
    Rounds a number to the first decimal place. Useful for computing errors.

    Parameters
    ----------
    x : float
        Number to round.

    Returns
    -------
    rounded_number : float
        Rounded number.

    """
    rounded_number = round(x, -int(floor(log10(abs(x)))))
    return rounded_number
            
def pretrain_loop(model, loss_fn, optimizer, train_data, q_2, arch, 
                  integration, w_i, norm_s, norm_d, psi_ansatz_s, 
                  psi_ansatz_d):
    """
    Pretraining loop.
    
    Parameters
    ----------   
    model : object
        Neural Network (torch) instance.
    loss_fn : function
        Loss function.
    optimizer : torch.optim
        Optimizer.
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
    psi_s_pred : tensor
        NN prediction for the S-state.
    psi_d_pred : tensor
        NN prediction for the D-state.
    k_s : tensor
        Overlap between the NN S-state ansatz and the analytical S-state ansatz.
    k_d : tensor
        Overlap between the NN D-state ansatz and the analytical D-state ansatz.

    """
    optimizer.zero_grad()
    (loss, psi_s_pred, psi_d_pred, 
     k_s, k_d) = loss_fn(model=model, 
                         arch=arch,
                         train_data=train_data,
                         norm_s = norm_s,
                         norm_d = norm_d,
                         q_2=q_2,
                         integration=integration,
                         w_i=w_i,
                         psi_ansatz_s=psi_ansatz_s,
                         psi_ansatz_d=psi_ansatz_d)
    loss.backward()
    optimizer.step()
    
    return psi_s_pred, psi_d_pred, k_s, k_d

def train_loop(model, loss_fn, optimizer, train_data, q_2, arch, integration, 
               w_i, q2_128, wi_128, V_ij, psi_exact_s, psi_exact_s_norm,
               psi_exact_d, psi_exact_d_norm):
    """
    Training loop. 

    Parameters
    ----------
    model : object
        Neural Network (torch) instance.
    loss_fn : function
        Loss function.
    optimizer : torch.optim
        Optimizer.
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
    loss : tensor
        Loss function value.
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
    
    optimizer.zero_grad()
    (loss, ann_s, ann_d, norm2, 
     K, U, E, ks, kd, pd) = loss_fn(model=model,
                                    train_data=train_data, 
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
    loss.backward()
    optimizer.step()
    
    return loss, ann_s, ann_d, norm2, K, U, ks, kd, pd

def f(E_accum, ks_accum, kd_accum, pd_accum, print_stats_for_every_model,
      mean_E, mean_ks, mean_kd, mean_PD):  
    """
    Computes top and bottom errors of input values, and prints this information
    to console.

    Parameters
    ----------
    E_accum : list
        Contains an energy value for each iteration.
    ks_accum : list
        Contains an S-state fidelity value for each iteration.
    kd_accum : list
        Contains a D-state fidelity value for each iteration.
    pd_accum : list
        Contains a D-state probability for each iteration.
    print_stats_for_every_model : boolean
        Whether information is printed out in console or not.
    mean_E : float
        Mean energy.
    mean_ks : float
        Mean S-state fidelity.
    mean_kd : float
        Mean of the D-state fidelity.
    mean_PD : float
        Mean of the D-state probabilities.

    Returns
    -------
    mean_E_top : float
        Mean of the energy upper bounds.
    mean_E_bot : float
        Mean of the energy lower bounds.
    mean_ks_top : float
        Mean of the S-state fidelity upper bounds.
    mean_ks_bot : float
        Mean of the S-state fidelity lower bounds.
    mean_kd_top : float
        Mean of the D-state fidelity upper bounds.
    mean_kd_bot : float
        Mean of the D-state fidelity lower bounds.
    mean_pd_top : float
        Mean of the D-state probability upper bounds.
    mean_pd_bot : float
        Mean of the D-state probability lower bounds.

    """     
    
    mean_E_top = max(E_accum)
    mean_E_bot = min(E_accum)
    
    mean_ks_top = max(ks_accum)
    mean_ks_bot = min(ks_accum)
    
    mean_kd_top = max(kd_accum)
    mean_kd_bot = min(kd_accum)
    
    mean_pd_top = max(pd_accum)
    mean_pd_bot = min(pd_accum)
               
    if print_stats_for_every_model == True:
        E_plus = round_to_1(abs(abs(mean_E_top)-abs(mean_E)))
        E_plus_error = mean_E+round_to_1(abs(abs(mean_E_top)-abs(mean_E)))
        E_minus = round_to_1(abs(abs(mean_E_bot)-abs(mean_E)))
        E_minus_error = mean_E-round_to_1(abs(abs(mean_E_bot)-abs(mean_E)))
        Ks_plus = round_to_1(abs(abs(mean_ks_top)-abs(mean_ks)))
        Ks_minus = round_to_1(abs(abs(mean_ks_bot)-abs(mean_ks)))
        Kd_plus = round_to_1(abs(abs(mean_kd_top)-abs(mean_kd)))
        Kd_minus = round_to_1(abs(abs(mean_kd_bot)-abs(mean_kd)))
        Pd_plus = round_to_1(abs(abs(mean_pd_top)-abs(mean_PD)))
        Pd_minus = round_to_1(abs(abs(mean_pd_bot)-abs(mean_PD)))
        print("\nE = {:16.6f}".format(mean_E))
        print("E+ = {}, {}".format(E_plus, E_plus_error))
        print("E- = {}, {}".format(E_minus, E_minus_error))
        
        print("\nKs = {:16.6f}".format(mean_ks))
        print("Ks+ = {}".format(Ks_plus))
        print("Ks- = {}".format(Ks_minus))
        
        print("\nKd = {:16.6f}".format(mean_kd))
        print("Kd+ = {}".format(Kd_plus))
        print("Kd- = {}".format(Kd_minus))
        
        print("\nPd = {:16.6f}".format(mean_PD))
        print("Pd+ = {}".format(Pd_plus))
        print("Pd- = {}".format(Pd_minus))
    
    return (mean_E_top, mean_E_bot, mean_ks_top, mean_ks_bot, mean_kd_top,
            mean_kd_bot, mean_pd_top, mean_pd_bot)

def error_update(E_accum, mean_E_accum, E_top_accum, E_bot_accum, 
                 mean_E_top, mean_E_bot, mean_ks_top, mean_ks_bot, 
                 mean_kd_top, mean_kd_bot, mean_pd_top, mean_pd_bot, 
                 mean_ks, ks_accum, 
                 mean_ks_accum, ks_top_accum, ks_bot_accum, kd_accum,
                 mean_kd, mean_kd_accum, kd_top_accum, kd_bot_accum, pd_accum,
                 mean_PD, mean_PD_accum, pd_top_accum, pd_bot_accum):
    """
    Updates the different error quantities with information from the current
    iteration.

    Parameters
    ----------
    E_accum : list
        Contains a value of the energy for every iteration.
    mean_E_accum : list
        Contains the mean of the list 'E_accum'. At first it is an empty list.
    E_top_accum : list
        Contains an energy upper bound for every iteration.
    E_bot_accum : list
        Contains an energy lower bound for every iteration.
    mean_E_top : float
        Mean of the energy upper bounds.
    mean_E_bot : float
        Mean of the energy lower bounds.
    mean_ks_top : float
        Mean of the S-state fidelity upper bounds.
    mean_ks_bot : float
        Mean of the S-state fidelity lower bounds.
    mean_kd_top : float
        Mean of the D-state fidelity upper bounds.
    mean_kd_bot : float
        Mean of the D-state fidelity lower bounds.
    mean_pd_top : float
        Mean of the D-state probability upper bounds.
    mean_pd_bot : float
        Mean of the D-state probability lower bounds.
    mean_ks : float
        Mean S-state fidelity.
    ks_accum : list
        Contains a value of the S-state fidelity for every iteration.
    mean_ks_accum : list
        Contains the mean of the list 'ks_accum'. At first it is an empty list.
    ks_top_accum : list
        Contains an S-state fidelity upper bound for every iteration.
    ks_bot_accum : list
        Contains an S-state fidelity lower bound for every iteration.
    kd_accum : list
        Contains a value of the D-state fidelity for every iteration.
    mean_kd : float
        Mean of the D-state fidelity.
    mean_kd_accum : list
        Contains the mean of the list 'kd_accum'. At first it is an empty list.
    kd_top_accum : list
        Contains an D-state fidelity upper bound for every iteration.
    kd_bot_accum : list
        Contains an D-state fidelity lower bound for every iteration.
    pd_accum : list
        Contains a value of the D-state probability for every iteration.
    mean_PD : float
        Mean of the D-state probabilities.
    mean_PD_accum : list
        Contains the mean of the list 'pd_accum'. At first it is an empty list.
    pd_top_accum : list
        Contains an D-state probability upper bound for every iteration.
    pd_bot_accum : list
        Contains an D-state probability lower bound for every iteration.

    Returns
    -------
    mean_E : float
        Mean of the list 'E_accum'.
    mean_E_accum : list
        Contains the mean of the list 'E_accum'. At first it is an empty list.
    E_top_accum : list
        Contains an energy upper bound for every iteration.
    E_bot_accum : list
        Contains an energy lower bound for every iteration.
    mean_E_top : float
        Mean of the energy upper bounds.
    mean_E_bot : float
        Mean of the energy lower bounds.
    mean_ks : float
        Mean S-state fidelity.
    mean_ks_accum : list
        Contains the mean of the list 'ks_accum'. At first it is an empty list.
    ks_top_accum : list
        Contains an S-state fidelity upper bound for every iteration.
    ks_bot_accum : list
        Contains an S-state fidelity lower bound for every iteration.
    mean_ks_top : float
        Mean of the S-state fidelity upper bounds.
    mean_ks_bot : float
        Mean of the S-state fidelity lower bounds.
    mean_kd : float
        Mean of the D-state fidelity.
    mean_kd_accum : list
        Contains the mean of the list 'kd_accum'. At first it is an empty list.
    kd_top_accum : list
        Contains an D-state fidelity upper bound for every iteration.
    kd_bot_accum : list
        Contains an D-state fidelity lower bound for every iteration.
    mean_kd_top : float
        Mean of the D-state fidelity upper bounds.
    mean_kd_bot : float
        Mean of the D-state fidelity lower bounds.
    mean_PD : float
        Mean of the D-state probabilities.
    mean_PD_accum : list
        Contains the mean of the list 'pd_accum'. At first it is an empty list.
    pd_top_accum : list
        Contains an D-state probability upper bound for every iteration.
    pd_bot_accum : list
        Contains an D-state probability lower bound for every iteration.
    mean_pd_top : float
        Mean of the D-state probability upper bounds.
    mean_pd_bot : float
        Mean of the D-state probability lower bounds.
    break_ : Boolean
        Whether to break out of the loop in the main program or not. The 
        condition for breaking is having 'mean_E' <= 0.

    """

    break_ = False
    # Energy
    mean_E = sum(E_accum)/len(E_accum)
    mean_E_accum.append(mean_E)
    candidate_to_extreme = E_accum[-4]
    if E_accum[-2]>E_accum[-1] and E_accum[-2]>E_accum[-3]:
        E_top_accum.append(candidate_to_extreme)
    elif E_accum[-2]<E_accum[-1] and E_accum[-2]<E_accum[-3]:
        E_bot_accum.append(candidate_to_extreme)
    if len(E_top_accum) > 0:
        mean_E_top = sum(E_top_accum)/len(E_top_accum)
    if len(E_bot_accum) > 0:
        mean_E_bot = sum(E_bot_accum)/len(E_bot_accum)
        
    if mean_E <= 0.:
        # Overlap S
        mean_ks = sum(ks_accum)/len(ks_accum)
        mean_ks_accum.append(mean_ks)
        candidate_to_extreme = ks_accum[-2]
        if ks_accum[-2]>ks_accum[-1] and ks_accum[-2]>ks_accum[-3]:
            ks_top_accum.append(candidate_to_extreme)
        elif ks_accum[-2]<ks_accum[-1] and ks_accum[-2]<ks_accum[-3]:
            ks_bot_accum.append(candidate_to_extreme)
        if len(ks_top_accum) > 0:
            mean_ks_top = sum(ks_top_accum)/len(ks_top_accum)
        if len(ks_bot_accum) > 0:
            mean_ks_bot = sum(ks_bot_accum)/len(ks_bot_accum)
            
        # Overlap D
        mean_kd = sum(kd_accum)/len(kd_accum)
        mean_kd_accum.append(mean_kd)
        candidate_to_extreme = kd_accum[-2]
        if kd_accum[-2]>kd_accum[-1] and kd_accum[-2]>kd_accum[-3]:
            kd_top_accum.append(candidate_to_extreme)
        elif kd_accum[-2]<kd_accum[-1] and kd_accum[-2]<kd_accum[-3]:
            kd_bot_accum.append(candidate_to_extreme)
        if len(kd_top_accum) > 0:
            mean_kd_top = sum(kd_top_accum)/len(kd_top_accum)
        if len(kd_bot_accum) > 0:
            mean_kd_bot = sum(kd_bot_accum)/len(kd_bot_accum)        

        # Prob. of D_state
        mean_PD = sum(pd_accum)/len(pd_accum)
        mean_PD_accum.append(mean_PD)
        candidate_to_extreme = pd_accum[-2]
        if pd_accum[-2]>pd_accum[-1] and pd_accum[-2]>pd_accum[-3]:
            pd_top_accum.append(candidate_to_extreme)
        elif pd_accum[-2]<pd_accum[-1] and pd_accum[-2]<pd_accum[-3]:
            pd_bot_accum.append(candidate_to_extreme)
        if len(pd_top_accum) > 0:
            mean_pd_top = sum(pd_top_accum)/len(pd_top_accum)
        if len(pd_bot_accum) > 0:
            mean_pd_bot = sum(pd_bot_accum)/len(pd_bot_accum)
    else: 
        # print('Skipping non-convergent model...')
        break_ = True
        
    return (mean_E, mean_E_accum, E_top_accum, E_bot_accum, mean_E_top, 
            mean_E_bot, mean_ks, mean_ks_accum, ks_top_accum, ks_bot_accum, 
            mean_ks_top, mean_ks_bot, mean_kd, mean_kd_accum, kd_top_accum, 
            kd_bot_accum, mean_kd_top, mean_kd_bot, mean_PD, mean_PD_accum, 
            pd_top_accum, pd_bot_accum, mean_pd_top, mean_pd_bot, break_)