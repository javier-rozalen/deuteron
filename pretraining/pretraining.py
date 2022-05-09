# -*- coding: utf-8 -*-
import sys
###################################### IMPORTANT INFORMATION ######################################
"""
This program (pre)trains an Artificial Neural Network to match a given function. The ANN  
has one input and two outputs, and hidden layers with Nhid hidden neurons in total. A  
64-points (preferentially) tangential mesh is used as the momentum space upon which we train the ANN. 

The models are saved in the saved_models/pretraining/ folder.

Parameters to adjust manually:
    GENERAL PARAMETERS
    network_arch --> '1sc', '2sc', '1sd', '2sd'. Network architecture
    device --> 'cpu', 'cuda', Physical processing unit into which we load our model.
    nchunks --> int, Number of batches of pretrained models.
                This is only for parallelized computing. If using a single instance of this program,
                set nchunks to 1. 
    which_chunk --> int, Selects the desired batch (by number). If using multiple instances of this
                    program, set this parameter to 'int(sys.argv[1])', for example.
    save_model --> Boolean, Saves the model at the end of the training or not.
    save_plot --> Boolean, Saves the plot at the end of the training or not.
    epochs --> int, Number of training epochs. 
    periodic_plots --> Boolean, Sets whether plots of the wave function are periodically shown.
    show_arch --> Boolean, Sets whether the network architecture is printed at the beginning of each
                  file.
    leap --> int, Number of epochs between updates and/or plots.
    hidden_neurons --> list, List of the number of hidden neurons we want to pretrain.
    seed_from --> int, Initial seed.
    seed_to --> int, Final seed.
    
    MESH PARAMETERS 
    q_max --> float, Maximum value of the momentum axis
    n_samples --> int, Number of mesh points. DO NOT CHANGE THE DEFAULT VALUE (64).
    n_test --> int, Number of test mesh points.
    train_a --> float, Lower limit of momentum
    train_b --> float, Upper limit of momentum. It is set to q_max by default.
    test_a --> float, Lower limit of test momentum.
    test_b --> float, Upper limit of test momentum. 
    
    TRAINING HYPERPARAMETERS
    learning_rate --> float, Learning rate. Must be specified in decimal notation.
    epsilon --> float, It appears in RMSProp and other optimizers
    smoothing_constant --> float, It appears in RMSProp and other optimizers
    momentum --> float, Momentum of the optimizer
"""

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
network_arch = '1sc'
device = 'cpu' 
nchunks = 1
which_chunk = 4 if nchunks != 1 else 0
save_model = True  
save_plot = True
epochs = 2000
periodic_plots = False
show_arch = False
leap = 500
hidden_neurons = [20,30,40,60,80,100] if network_arch != '2sd' else [20,32,40,60,80,100]
seed_from = 1
seed_to = 21

# Mesh parameters
q_max = 500
n_samples = 64 # Do not change this.
n_test = n_samples 
train_a = 0
train_b = q_max
test_a = 0
test_b = 5

# Training hyperparameters
learning_rate = 0.01 # Use decimal notation 
epsilon = 1e-8
smoothing_constant = 0.9
momentum = 0.

###################################### IMPORTS ######################################
import pathlib,os
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

import torch, time, math
import numpy as np
from tqdm import tqdm

# My modules
from modules.physical_constants import *
import modules.integration as integration
import modules.neural_networks as neural_networks
from modules.plotters import pretraining_plots

###################################### MISC. ######################################

chunk_size = int(len(hidden_neurons)/nchunks)
print(f'\nTraining model {network_arch}.')
net_arch_name_map = {'1sc':'fully_connected_ann','2sc':'fully_connected_ann',
                     '1sd':'separated_ann','2sd':'separated_ann'}

# Directory management and creation
path_steps_models = ['..','saved_models','pretraining',f'nlayers{network_arch[0]}',f'{net_arch_name_map[network_arch]}',
              'Sigmoid',f'lr{learning_rate}','models']
path_steps_plots = ['..','saved_models','pretraining',f'nlayers{network_arch[0]}',f'{net_arch_name_map[network_arch]}',
              'Sigmoid',f'lr{learning_rate}','plots']
for i in range(len(path_steps_models)):
    potential_dir_models = '/'.join(path_steps_models[:i+1])
    potential_dir_plots = '/'.join(path_steps_plots[:i+1])
    if not os.path.exists(potential_dir_models):
        os.makedirs(potential_dir_models)
        print(f'Creating directory {potential_dir_models}.')
    if not os.path.exists(potential_dir_plots):
        os.makedirs(potential_dir_plots)
        print(f'Creating directory {potential_dir_plots}.')
    

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
Q_test = Q_train

###################################### TARGET FUNCTION ######################################
psi_ansatz_s = torch.exp((-1.5**2)*Q_train**2/2) # target function L=0
psi_ansatz_d = (Q_train**2)*torch.exp((-1.5**2)*Q_train**2/2) # target function L=2

norm_s = integration.gl64(w_i,q_2*(psi_ansatz_s**2))
norm_d = integration.gl64(w_i,q_2*(psi_ansatz_d**2))

psi_ansatz_s_normalized = psi_ansatz_s/torch.sqrt(norm_s)
psi_ansatz_d_normalized = psi_ansatz_d/torch.sqrt(norm_d)


##################################### LOOP OVER HIDDEN NEURONS,SEEDS MODELS #####################################
hn_number = 0
start_time_all = time.time()
for j in chunks(hidden_neurons,chunk_size)[which_chunk]:
    """"We iterate over all the hidden neurons numbers."""
    for i in range(seed_from,seed_to+1):
        """"We iterate over all seeds for a given Nhid."""
        
        seed,Nhid = i,j
        print("\nNeurons = {}, Seed = {}/{}".format(j,seed,seed_to))
        torch.manual_seed(seed)
    
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
        if show_arch:
            print("NN architecture:\n",psi_ann)
            neural_networks.show_layers(psi_ann)
        
        ################### COST FUNCTION ###################
        def cost():
            """Returns the cost, which is computed from the overlap between the network"""                      
            global psi_s_pred,psi_d_pred
            ann = psi_ann(Q_train.clone().unsqueeze(1))
            if network_arch[2] == 'c':
                ann1,ann2 = ann[:,:1].squeeze(),ann[:,1:].squeeze()
            else:
                ann1,ann2 = ann[0],ann[1]
            
            global k_s
            num = ann1*psi_ansatz_s*q_2
            denom1 = q_2*(ann1**2) 
            nume = (integration.gl64(w_i,num))**2
            den1 = integration.gl64(w_i,denom1)
            den2 = norm_s
            k_s = nume/(den1*den2)
            c_s = (k_s-torch.tensor(1.))**2
            norm_s_ann = den1
            
            # C(L=2)
            global k_d
            num = ann2*psi_ansatz_d*q_2
            denom1 = q_2*(ann2**2) 
            nume = (integration.gl64(w_i,num))**2
            den1 = integration.gl64(w_i,denom1)
            den2 = norm_d
            k_d = nume/(den1*den2)
            c_d = (k_d-torch.tensor(1.))**2
            norm_d_ann = den1
            
            psi_s_pred = ann1/torch.sqrt(norm_s_ann)
            psi_d_pred = ann2/torch.sqrt(norm_d_ann)
            
            return c_s+c_d
          
        ################################ EPOCH LOOP ##################################
        loss_fn = cost
        optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,eps=epsilon,
                                        alpha=smoothing_constant,momentum=momentum)
        
        def train_loop(loss_fn,optimizer):
            """Trains the Neural Network over the training dataset"""    
            optimizer.zero_grad()
            loss_fn().backward()
            optimizer.step()
                    
        sep = "("
        path_model = "../saved_models/pretraining/nlayers{}/{}/Sigmoid/lr{}/models/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}".format(network_arch[0],net_arch_name_map[network_arch],learning_rate,Nhid,epochs,seed,learning_rate,
                                                                                      str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                      str(optimizer).split(sep,1)[0].strip())
        path_plot = "../saved_models/pretraining/nlayers{}/{}/Sigmoid/lr{}/plots/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}".format(network_arch[0],net_arch_name_map[network_arch],learning_rate,Nhid,epochs,seed,learning_rate,
                                                                                      str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                      str(optimizer).split(sep,1)[0].strip())
        
        saved_models_dir = f'../saved_models/pretraining/nlayers{network_arch[0]}/{net_arch_name_map[network_arch]}/Sigmoid/lr{learning_rate}/models/'
        name_without_dirs = 'nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}.pt'.format(Nhid,epochs,seed,learning_rate,
                                                                                      str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                      str(optimizer).split(sep,1)[0].strip())

        if name_without_dirs not in os.listdir(saved_models_dir): 
            # We store the energy data in lists for later plotting
            overlap_s,overlap_d = [],[]
            
            start_time = time.time()
            for t in tqdm(range(epochs)):  
                train_loop(loss_fn,optimizer)
                overlap_s.append(k_s.item())
                overlap_d.append(k_d.item())
            
                if ((t+1)%leap)==0:
                    if periodic_plots:
                        pretraining_plots(Q_test,psi_s_pred,psi_d_pred,n_test,n_samples,psi_ansatz_s_normalized,psi_ansatz_d_normalized,
                                              overlap_s,overlap_d,path_plot,t,False) 
            
                if t == epochs-1:
                    exec_time = "{:2.0f}".format(time.time()-start_time)
                    pretraining_plots(Q_test,psi_s_pred,psi_d_pred,n_test,n_samples,psi_ansatz_s_normalized,psi_ansatz_d_normalized,
                                          overlap_s,overlap_d,path_plot,t,save_plot,exec_time)
                    
            print('Model pretrained!')
            print('Total execution time:  {:6.2f} seconds (run on {})'.format(time.time()-start_time_all,device))
            
            full_path_model = '{}.pt'.format(path_model)
            full_path_plot = '{}.png'.format(path_plot)
            
            if save_model == True:
                state_dict = {'model_state_dict':psi_ann.state_dict(),
                              'optimizer_state_dict':optimizer.state_dict()}
                torch.save(state_dict,full_path_model)
                print(f'Model saved in {full_path_model}')
            if save_plot: 
                print('Plot saved in {}'.format(path_plot))
            
        else:
            print(f'Skipping already pretrained model {name_without_dirs}...')
            
        hn_number += 1
    
print("\nAll done! :)")
   