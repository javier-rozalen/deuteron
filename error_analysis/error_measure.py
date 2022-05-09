# -*- coding: utf-8 -*-

###################################### IMPORTANT INFORMATION ######################################
"""
This program runs the error analysis of the (fully) trained models. It computes the mean values of 
the energy, the fidelity and the probability of the D-state, along with their associated errors. The
details can be found in the main text of the article. 

The data is saved under the error_analysis/ folder.

Parameters to adjust manually:
    GENERAL PARAMETERS
    network_arch --> '1sc', '2sc', '1sd', '2sd'. Network architecture
    device --> 'cpu', 'cuda', Physical processing unit into which we load our model.
    nchunks --> int, Number of batches of pretrained models.
                This is only for parallelized computing. If using a single instance of this program,
                set nchunks to 1. 
    which_chunk --> int, Selects the desired batch (by number). If using multiple instances of this
                    program, set this parameter to 'int(sys.argv[1])', for example.
    save_data --> Boolean, Saves the data at the end of the analysis or not.
    epochs --> int, Number of training epochs. 
    periodic_plots --> Boolean, Sets whether plots of the wave function are periodically shown.
    show_arch --> Boolean, Sets whether the network architecture is printed at the beginning of each
                  file.
    leap --> int, Number of epochs between updates and/or plots.
    print_stats_for_every_model --> Boolean, The name is self-explanatory.
    
    MESH PARAMETERS 
    q_max --> float, Maximum value of the momentum axis
    n_samples --> int, Number of mesh points. DO NOT CHANGE THE DEFAULT VALUE (64).
    n_test --> int, Number of test mesh points.
    train_a --> float, Lower limit of momentum
    train_b --> float, Upper limit of momentum. It is set to q_max by default.
    test_a --> float, Lower limit of test momentum.
    test_b --> float, Upper limit of test momentum. 
    
    ADAPTIVE PLOTTING PARAMETERS
    adaptive_lims --> Boolean, Sets whether the plot limits are adaptive or not.
    factor_E_sup --> Float, Scale factor of E that sets the E plot upper limit.
    factor_E_inf --> Float, Scale factor of E thta sets the E plot lower limit.
    factor_k_sup --> Float, Idem. 
    factor_k_inf --> Float, Idem.
    factor_pd_sup --> Float, Idem.
    factor_pd_inf --> Float, Idem.
    
    TRAINING HYPERPARAMETERS
    learning_rate --> float, Learning rate. Must be specified in decimal notation.
    epsilon --> float, It appears in RMSProp and other optimizers
    smoothing_constant --> float, It appears in RMSProp and other optimizers
    momentum --> float, Momentum of the optimizer
"""

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
network_arch = '1sc'
device='cpu' 
nchunks = 1
which_chunk = 0
save_data = True
epochs = 300
periodic_plots = True
show_arch = False
leap = epochs
print_stats_for_every_model = False

# Mesh parameters
q_max = 500
n_samples = 64 # Do not change this.
n_test = n_samples 
train_a = 0
train_b = q_max
test_a = 0
test_b = 5

# Adaptive plotting parameters
adaptive_lims = True
factor_E_sup = 1.00001
factor_E_inf = 1.001
factor_k_sup = 1.001
factor_k_inf = 0.99997
factor_pd_sup = 1.001
factor_pd_inf = 0.999

# Training hyperparameters
learning_rate = 0.01 # use decimal notation
epsilon = 1e-8
smoothing_constant = 0.9
momentum= 0.

###################################### IMPORTS ######################################
import pathlib, os,sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

import torch, time, math, re
import numpy as np
from math import log10, floor
from tqdm import tqdm

# My modules
from modules.physical_constants import *
import modules.integration as integration
import modules.neural_networks as neural_networks
import modules.N3LO as N3LO
from modules.plotters import error_measure_plots

net_arch_name_map = {'1sc':'fully_connected_ann','2sc':'fully_connected_ann',
                     '1sd':'separated_ann','2sd':'separated_ann'}
path_to_trained_models = f'../saved_models/n3lo/nlayers{network_arch[0]}/{net_arch_name_map[network_arch]}/Sigmoid/lr{learning_rate}/models/'
list_of_trained_models = os.listdir(path_to_trained_models)
chunk_size = int(len(list_of_trained_models)/nchunks)

# We sort the list of pretrained models by nhid
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]
list_of_trained_models.sort(key=num_sort)

# We split the list into chunks of the desired size to seize multi-core PCs
def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

###################################### MESH PREPARATION ######################################
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

# Integration-specific
x_i_int = [torch.tensor(float(e)) for e in x]
w_i_int = [torch.tensor(float(e)) for e in w]

x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # q mesh from 0 to 1
k = [(q_max/math.tan(x_i[-1]*math.pi/2))*math.tan(float(e)*math.pi/2) for e in x_i] # tangential q mesh from 0 to q_max
w_i = [torch.tensor(float(e)/2) for e in w] # GL weights
cos2 = [1/(math.cos(float(x_i[i])*math.pi/2))**2 for i in range(n_samples)]
p = (q_max/math.tan(x_i[-1]*math.pi/2))*math.pi/2 
w_i = [p*w_i[r]*cos2[r] for r in range(n_samples)]
w_i = torch.stack(w_i)
Q_train = torch.tensor(k) # Momentum mesh
q_2 = Q_train**2 # Squared momentum mesh

###################################### BENCHMARK DATA FETCHING/COMPUTATION ######################################
# N3LO potential
n3lo = N3LO.N3LO("../deuteron_data/2d_vNN_J1.dat","../deuteron_data/wfk.dat")
V_ij = n3lo.getPotential()

# Target wavefunction (A.Rios, James W.T. Keeble)
psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)   
psi_target_s_norm = integration.gl64(q_2*(psi_target_s)**2,w_i)
psi_target_d_norm = integration.gl64(q_2*(psi_target_d)**2,w_i)
psi_target_s_normalized = psi_target_s/torch.sqrt(psi_target_s_norm+psi_target_d_norm)
psi_target_d_normalized = psi_target_d/torch.sqrt(psi_target_d_norm+psi_target_s_norm)

# Vectors for the cost function
q2_128 = torch.cat((q_2,q_2))
wi_128 = torch.cat((w_i,w_i))

# Exact diagonalization wavefunction
psi_exact_s = []
psi_exact_d = []
Q_train1 = []
with open('../deuteron_data/wfk_exact_diagonalization.txt','r') as file:
    c = 0
    for line in file.readlines():
        if c >= 2:
            psi_exact_s.append(torch.tensor(float(line.split(' ')[4])))
            psi_exact_d.append(torch.tensor(float(line.split(' ')[6])))
            Q_train1.append(torch.tensor(float(line.split(' ')[2])))
        c += 1

psi_exact_s = torch.stack(psi_exact_s)
psi_exact_d = torch.stack(psi_exact_d)
psi_exact_s_norm = integration.gl64(q_2*(psi_exact_s)**2,w_i)
psi_exact_d_norm = integration.gl64(q_2*(psi_exact_d)**2,w_i)
psi_exact_s_normalized = psi_exact_s/torch.sqrt(psi_exact_s_norm+psi_exact_d_norm)
psi_exact_d_normalized = psi_exact_d/torch.sqrt(psi_exact_s_norm+psi_exact_d_norm)

# Energy obtained with the exact diagonalization wf
def get_energy_pd():
    """Returns the energy and probability of D-state of the exact diagonalization wf"""
    
    psi_128 = torch.cat((psi_exact_s_normalized,psi_exact_d_normalized))
    y = psi_128*q2_128*wi_128
    U_exact = torch.matmul(y,torch.matmul(V_ij,y)) 
    K_exact = ((hbar**2)/mu)*torch.dot(wi_128,(psi_128*q2_128)**2)
    E_exact = K_exact + U_exact
    PD_exact = 100.*psi_exact_d_norm/(psi_exact_s_norm+psi_exact_d_norm)
    return E_exact,PD_exact

E_exact,PD_exact = get_energy_pd()[0],get_energy_pd()[1]

##################################### LOOP TRAINED MODELS #####################################
# We iterate over all the pretrained models
start_time_total = time.time()
for model in tqdm(iterable=chunks(list_of_trained_models,chunk_size)[which_chunk]): 
    print('\nAnalyzing file: {}'.format(model))
    seed = int(model.split('_')[2].replace('seed',''))
        
    # Single-layer specific params
    Nhid = int(model.split('_')[0].replace('nhid',''))
           
    # ANN Parameters
    if network_arch == '1sc':
        Nhid_prime = Nhid
    elif network_arch == '2sc' or network_arch == '1sd':
        Nhid_prime = int(Nhid/2)
    elif network_arch == '2sd':
        Nhid_prime = int(Nhid/4)
        
    Nin = 1
    Nout = 1 if network_arch == '2sd' else 2 
    W1 = torch.rand(Nhid_prime,Nin,requires_grad=True)*(-1.) 
    B = torch.rand(Nhid_prime)*2.-torch.tensor(1.) 
    W2 = torch.rand(Nout,Nhid_prime,requires_grad=True) 
    Ws2 = torch.rand(1,Nhid_prime,requires_grad=True) 
    Wd2 = torch.rand(1,Nhid_prime,requires_grad=True) 
    
    # We load our psi_ann to the CPU (or GPU)
    net_arch_map = {'1sc':neural_networks.sc_1,'2sc':neural_networks.sc_2,
                    '1sd':neural_networks.sd_1,'2sd':neural_networks.sd_2}
    psi_ann = net_arch_map[network_arch](Nin,Nhid_prime,Nout,W1,Ws2,B,W2,Wd2).to(device)
    psi_ann.load_state_dict(torch.load(f'{path_to_trained_models}{model}')['model_state_dict'])
    
    if show_arch:
        print("NN architecture:\n",psi_ann)
        neural_networks.show_layers(psi_ann)
    
    ############################ COST FUNCTION ###################################
    def cost():
        """Returns the cost computed as the expected energy using the current wavefunction (ANN)
        Also returns the overlap with the theoretical wavefunction"""
        # Wavefunction 
        global ann_s,ann_d,norm2,K,U,E,ks,kd,pd
        
        ann = psi_ann(Q_train.clone().unsqueeze(1))
        if network_arch[2] == 'c':
            ann1,ann2 = ann[:,:1].squeeze(),ann[:,1:].squeeze()
        else:
            ann1,ann2 = ann[0],ann[1]
        
        # Norm 
        norm_s = integration.gl64(w_i,q_2*(ann1)**2)
        norm_d = integration.gl64(w_i,q_2*(ann2)**2)    
        norm2 = norm_s+norm_d
        ann_s = ann1/torch.sqrt(norm2)
        ann_d = ann2/torch.sqrt(norm2)
        pd = 100.*norm_d/norm2
        
        # S-D concatenated wavefunction
        psi_128 = torch.cat((ann_s,ann_d)) 
        
        # K
        K = ((hbar**2)/mu)*torch.dot(wi_128,(psi_128*q2_128)**2)
        
        # U
        y = psi_128*q2_128*wi_128
        U = torch.matmul(y,torch.matmul(V_ij,y))
    
        # E
        E = K+U
        
        # Overlap (L=0)
        ks = ((integration.gl64(w_i,q_2*ann1*psi_exact_s))**2)/(norm_s*psi_exact_s_norm)
        
        # Overlap (L=2)
        kd = ((integration.gl64(w_i,q_2*ann2*psi_exact_d))**2)/(norm_d*psi_exact_d_norm)
        
        return E
    
    ######################### TRAINING AND TESTING ###############################
    # Training parameters
    loss_fn = cost
    optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,eps=epsilon,
                                    alpha=smoothing_constant,momentum=momentum)
    optimizer.load_state_dict(torch.load('{}{}'.format(path_to_trained_models,model))['optimizer_state_dict'])
    
    def train_loop(loss_fn,optimizer):
        """Trains the Neural Network over the training dataset"""    
        optimizer.zero_grad()
        loss_fn().backward()
        optimizer.step()
        
    ################################ EPOCH LOOP ##################################
    # We store the energy data in lists so as to plot it
    K_accum,U_accum,E_accum,ks_accum,kd_accum,pd_accum=[],[],[],[],[],[]
    
    def round_to_1(x):
        return round(x, -int(floor(log10(abs(x)))))
    
    # Error analysis parameters
    mean_E_accum,E_top_accum,E_bot_accum,mean_ks_accum,mean_kd_accum,ks_top_accum,ks_bot_accum=[],[],[],[],[],[],[]
    kd_top_accum,kd_bot_accum,mean_PD_accum,pd_top_accum,pd_bot_accum=[],[],[],[],[]
    mean_E,mean_E_top,mean_E_bot,mean_ks,mean_kd,mean_ks_top,mean_ks_bot,mean_kd_top,mean_kd_bot=0,0,0,0,0,0,0,0,0
    mean_PD,mean_pd_top,mean_pd_bot = 0,0,0
    
    def f():           
        global mean_E_top,mean_E_bot,mean_ks_top,mean_ks_bot,mean_kd_top,mean_kd_bot,mean_pd_top,mean_pd_bot
        mean_E_top = max(E_accum)
        mean_E_bot = min(E_accum)
        
        mean_ks_top = max(ks_accum)
        mean_ks_bot = min(ks_accum)
        
        mean_kd_top = max(kd_accum)
        mean_kd_bot = min(kd_accum)
        
        mean_pd_top = max(pd_accum)
        mean_pd_bot = min(pd_accum)
                   
        if print_stats_for_every_model == True:
            print("\nE = {:16.6f}".format(mean_E))
            print("E+ = {}".format(round_to_1(abs(abs(mean_E_top)-abs(mean_E)))))
            print("E- = {}".format(round_to_1(abs(abs(mean_E_bot)-abs(mean_E)))))
            
            print("\nKs = {:16.6f}".format(mean_ks))
            print("Ks+ = {}".format(round_to_1(abs(abs(mean_ks_top)-abs(mean_ks)))))
            print("Ks- = {}".format(round_to_1(abs(abs(mean_ks_bot)-abs(mean_ks)))))
            
            print("\nKd = {:16.6f}".format(mean_kd))
            print("Kd+ = {}".format(round_to_1(abs(abs(mean_kd_top)-abs(mean_kd)))))
            print("Kd- = {}".format(round_to_1(abs(abs(mean_kd_bot)-abs(mean_kd)))))
            
            print("\nPd = {:16.6f}".format(mean_PD))
            print("Pd+ = {}".format(round_to_1(abs(abs(mean_pd_top)-abs(mean_PD)))))
            print("Pd- = {}".format(round_to_1(abs(abs(mean_pd_bot)-abs(mean_PD)))))
        
        error_measure_plots(adaptive_lims,ks_accum,factor_k_sup,factor_k_inf,kd_accum,ks,kd,mean_ks,mean_ks_top,mean_ks_bot,
                        mean_kd,mean_kd_bot,mean_kd_top,factor_E_sup,factor_E_inf,E_accum,E,E_exact,mean_E,mean_E_top,
                        mean_E_bot,factor_pd_sup,factor_pd_inf,pd_accum,pd,PD_exact,mean_PD,mean_pd_top,mean_pd_bot,
                        periodic_plots)
        
    start_time = time.time()
    for t in range(epochs):        
        train_loop(loss_fn,optimizer)
        K_accum.append(K.item())
        U_accum.append(U.item())
        E_accum.append(E.item())
        ks_accum.append(ks.item())
        kd_accum.append(kd.item())
        pd_accum.append(pd.item())
                   
        ########################### ERROR ################################
        if t>3:
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
                print('Skipping non-convergent model...')
                break
            
        if mean_E <= 0.:
            if ((t+1)%leap)==0:
                if periodic_plots == True:
                    error_measure_plots(adaptive_lims,ks_accum,factor_k_sup,factor_k_inf,kd_accum,ks,kd,mean_ks,mean_ks_top,mean_ks_bot,
                                    mean_kd,mean_kd_bot,mean_kd_top,factor_E_sup,factor_E_inf,E_accum,E,E_exact,mean_E,mean_E_top,
                                    mean_E_bot,factor_pd_sup,factor_pd_inf,pd_accum,pd,PD_exact,mean_PD,mean_pd_top,mean_pd_bot,
                                    periodic_plots) 
        
            if t == epochs-1:
                exec_time = "{:2.0f}".format(time.time()-start_time)
                #error_measure_plots(adaptive_lims,ks_accum,factor_k_sup,factor_k_inf,kd_accum,ks,kd,mean_ks,mean_ks_top,mean_ks_bot,
                #                mean_kd,mean_kd_bot,mean_kd_top,factor_E_sup,factor_E_inf,E_accum,E,E_exact,mean_E,mean_E_top,
                #                mean_E_bot,factor_pd_sup,factor_pd_inf,pd_accum,pd,PD_exact,mean_PD,mean_pd_top,mean_pd_bot,
                #                periodic_plots)
                psi_s_pred = []
                psi_d_pred = []
        else:
            print('Skipping non-convergent model...')
            break
           
    if mean_E <= 0.:
        f()
    else:
        print('Skipping non-convergent model...')      
    
    if mean_E <= 0.:
        if save_data == True:
            # Directory management and creation
            path_steps_models = ['error_data',f'nlayers{network_arch[0]}',f'{net_arch_name_map[network_arch]}',
                          'Sigmoid',f'lr{learning_rate}']
            for i in range(len(path_steps_models)):
                potential_dir = '/'.join(path_steps_models[:i+1])
                if not os.path.exists(potential_dir):
                    os.makedirs(potential_dir)
                    print(f'Creating directory {potential_dir}.')
            filename=potential_dir+f'/nhid{Nhid}.txt' 
            with open(filename, 'a') as file:
                # E, E+, E-, Ks, Ks+, Ks-, Pd, Pd+, Pd-
                file.write(str(mean_E)+' '+
                            str(round_to_1(abs(abs(mean_E_top)-abs(mean_E))))+' '+
                            str(round_to_1(abs(abs(mean_E_bot)-abs(mean_E))))+' '+
                            str(mean_ks)+' '+
                            str(round_to_1(abs(abs(mean_ks_top)-abs(mean_ks))))+' '+
                            str(round_to_1(abs(abs(mean_ks_bot)-abs(mean_ks))))+' '+
                            str(mean_kd)+' '+
                            str(round_to_1(abs(abs(mean_kd_top)-abs(mean_kd))))+' '+
                            str(round_to_1(abs(abs(mean_kd_bot)-abs(mean_kd))))+' '+
                            str(mean_PD)+' '+
                            str(round_to_1(abs(abs(mean_pd_top)-abs(mean_PD))))+' '+
                            str(round_to_1(abs(abs(mean_pd_bot)-abs(mean_PD))))+' '+
                            str(seed)+' \n')
            file.close()
            print(f"\nData saved in {filename}.")
    else:
        print('Skipping non-convergent model...')
            
end_time_total = time.time()
print('\nAll done! :)')
print("\nTotal execution time: {:6.2f} seconds (run on {})".format(end_time_total-start_time_total,device))