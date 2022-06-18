# -*- coding: utf-8 -*-

###################################### IMPORTANT INFORMATION ######################################
"""
This program trains an Artificial Neural Network to find the deuteron wave function. The ANN  
has one input and two outputs, and hidden layers with Nhid hidden neurons in total. A  
64-points (preferentially) tangential mesh is used as the momentum space upon which we train the ANN. 
The potential used is N3LO.

The models are saved in the saved_models/n3lo/ folder.

Parameters to adjust manually:
    GENERAL PARAMETERS
    network_arch --> '1sc', '2sc', '1sd', '2sd'. Network architecture.
    device --> 'cpu', 'cuda', Physical processing unit into which we load our model.
    path_to_pretrained_model --> str, Path where the pretrained models are located.
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
    
    MESH PARAMETERS 
    q_max --> float, Maximum value of the momentum axis.
    n_samples --> int, Number of mesh points. DO NOT CHANGE THE DEFAULT VALUE (64).
    train_a --> float, Lower limit of momentum.
    train_b --> float, Upper limit of momentum. It is set to q_max by default.
    
    TRAINING HYPERPARAMETERS
    learning_rate --> float, Learning rate. Must be specified in decimal notation.
    epsilon --> float, It appears in RMSProp and other optimizers.
    smoothing_constant --> float, It appears in RMSProp and other optimizers.
    momentum --> float, Momentum of the optimizer.
"""

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
network_arch = '1sc'
device = 'cpu' 
nchunks = 1
which_chunk = 0 if nchunks != 1 else 0
save_model = False  
save_plot = False
epochs = 250000
periodic_plots = True
show_arch = False
leap = 5000

# Mesh parameters
q_max = 500
n_samples = 64 # Do not change this.
train_a = 0
train_b = q_max

# Training hyperparameters
learning_rate = 0.01 # Use decimal notation 
epsilon = 1e-8
smoothing_constant = 0.9
momentum = 0.

print(f'\nTraining model {network_arch}.')

###################################### IMPORTS ######################################
import pathlib, os, sys
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('.')

import torch, time, math, re
import numpy as np
from tqdm import tqdm

# My modules
from modules.physical_constants import *
import modules.N3LO as N3LO
import modules.integration as integration
import modules.neural_networks as neural_networks
from modules.plotters import minimisation_plots
from modules.dir_support import dir_support

###################################### PRETRAINED MODELS PREPARATION ######################################
net_arch_name_map = {'1sc':'fully_connected_ann','2sc':'fully_connected_ann',
                     '1sd':'separated_ann','2sd':'separated_ann'}

# Directory management and creation
dir_support(['saved_models','n3lo',f'nlayers{network_arch[0]}',f'{net_arch_name_map[network_arch]}',
              'Sigmoid',f'lr{learning_rate}','models'])
dir_support(['saved_models','n3lo',f'nlayers{network_arch[0]}',f'{net_arch_name_map[network_arch]}',
              'Sigmoid',f'lr{learning_rate}','plots'])
        
path_to_pretrained_model = '/'.join(['saved_models','pretraining',f'nlayers{network_arch[0]}',
                                     f'{net_arch_name_map[network_arch]}','Sigmoid',
                                     f'lr{learning_rate}','models'])+'/'
list_of_pretrained_models = os.listdir(path_to_pretrained_model)
chunk_size = int(len(list_of_pretrained_models)/nchunks)

# We sort the list of pretrained models by nhid
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]
list_of_pretrained_models.sort(key=num_sort)

# We split the list into chunks of the desired size to seize multi-core PCs
def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

###################################### MESH SET PREPARATION ######################################
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

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
n3lo = N3LO.N3LO("deuteron_data/2d_vNN_J1.dat","deuteron_data/wfk.dat")
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
with open('deuteron_data/wfk_exact_diagonalization.txt','r') as file:
    c = 0
    for line in file.readlines():
        if c >= 2:
            psi_exact_s.append(torch.tensor(float(line.split(' ')[4])))
            psi_exact_d.append(torch.tensor(float(line.split(' ')[6])))
            Q_train1.append(torch.tensor(float(line.split(' ')[2])))
        c += 1

psi_exact_s = torch.stack(psi_exact_s)
psi_exact_d = torch.stack(psi_exact_d)
psi_exact_s_norm = integration.gl64(w_i,q_2*(psi_exact_s)**2)
psi_exact_d_norm = integration.gl64(w_i,q_2*(psi_exact_d)**2)
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

##################################### LOOP OVER PRETRAINED MODELS #####################################
filenumber = 0
start_time_all = time.time()
for file in chunks(list_of_pretrained_models,chunk_size)[which_chunk]: 
    """"We iterate over all the pretrained models in the specified route above, and we extract
    the hyperparameters Nhid, seed, learning_rate from the name of the file."""
    
    name_of_file = file
    path_to_file = '{}{}'.format(path_to_pretrained_model,file)
    print('\nTraining on file {}'.format(file))
    print(f'File {filenumber+1}/{len(chunks(list_of_pretrained_models,chunk_size)[which_chunk])}')
    
    Nhid = int(name_of_file.split('_')[0].replace('nhid',''))
    seed = int(name_of_file.split('_')[2].replace('seed',''))
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
    psi_ann.load_state_dict(torch.load(path_to_file)['model_state_dict'])
    psi_ann.eval()
    if show_arch:
        print("NN architecture:\n",psi_ann)
        neural_networks.show_layers(psi_ann)
    
    ################### COST FUNCTION ###################
    def cost():
        """Returns the cost computed as the expected energy using the current wavefunction (ANN)
        Also returns the overlap with the theoretical wavefunction"""
        global ann_s,ann_d,norm2,K,U,E,ks,kd,pd

        ann = psi_ann(Q_train.clone().unsqueeze(1))
        if network_arch[2] == 'c':
            ann1,ann2 = ann[:,:1].squeeze(),ann[:,1:].squeeze()
        else:
            ann1,ann2 = ann[0],ann[1]

        # Norm
        norm_s = integration.gl64(q_2*(ann1)**2,w_i)
        norm_d = integration.gl64(q_2*(ann2)**2,w_i)    
        norm2 = norm_s+norm_d # squared norm
        
        # Wave function
        ann_s = ann1/torch.sqrt(norm2)
        ann_d = ann2/torch.sqrt(norm2)
        
        psi_128 = torch.cat((ann_s,ann_d)) # S-D concatenated wavefunction
        y = psi_128*q2_128*wi_128 # auxiliary tensor
        
        K = ((hbar**2)/mu)*torch.dot(wi_128,(psi_128*q2_128)**2) # Kinetic energy
        U = torch.matmul(y,torch.matmul(V_ij,y)) # U
        E = K+U # E
        ks = ((integration.gl64(q_2*ann1*psi_exact_s,w_i))**2)/(norm_s*psi_exact_s_norm) # Overlap (L=0)
        kd = ((integration.gl64(q_2*ann2*psi_exact_d,w_i))**2)/(norm_d*psi_exact_d_norm) # Overlap (L=2)
        pd = 100.*norm_d/norm2 # prob. of D-state
        
        return E
      
    ################################ EPOCH LOOP ##################################
    loss_fn = cost
    optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,eps=epsilon,
                                    alpha=smoothing_constant,momentum=momentum)
    optimizer.load_state_dict(torch.load(path_to_file)['optimizer_state_dict'])
    
    def train_loop(loss_fn,optimizer):
        """Trains the Neural Network over the training dataset"""    
        optimizer.zero_grad()
        loss_fn().backward()
        optimizer.step()
    
    sep = "("
    path_model = "saved_models/n3lo/nlayers{}/{}/Sigmoid/lr{}/models/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}".format(network_arch[0],net_arch_name_map[network_arch],learning_rate,Nhid,epochs,seed,learning_rate,
                                                                                  str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                  str(optimizer).split(sep,1)[0].strip())
    path_plot = "saved_models/n3lo/nlayers{}/{}/Sigmoid/lr{}/plots/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}".format(network_arch[0],net_arch_name_map[network_arch],learning_rate,Nhid,epochs,seed,learning_rate,
                                                                                  str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                  str(optimizer).split(sep,1)[0].strip())
    
    saved_models_dir = f'saved_models/n3lo/nlayers{network_arch[0]}/{net_arch_name_map[network_arch]}/Sigmoid/lr{learning_rate}/models/'
    name_without_dirs = 'nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}.pt'.format(Nhid,epochs,seed,learning_rate,
                                                                                  str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                                  str(optimizer).split(sep,1)[0].strip())

    if name_without_dirs not in os.listdir(saved_models_dir): 
        # We store the energy data in lists for later plotting
        K_accum,U_accum,E_accum,ks_accum,kd_accum,pd_accum=[],[],[],[],[],[]
        
        start_time = time.time()
        for t in tqdm(range(epochs)):        
            train_loop(loss_fn,optimizer)
            K_accum.append(K.item())
            U_accum.append(U.item())
            E_accum.append(E.item())
            ks_accum.append(ks.item())
            kd_accum.append(kd.item())
            pd_accum.append(pd.item())
        
            if ((t+1)%leap)==0:
                if periodic_plots:
                    minimisation_plots(k,ann_s,ann_d,psi_exact_s_normalized,psi_exact_d_normalized,ks_accum,kd_accum,
                                       ks,kd,K_accum,K,U,E,U_accum,E_accum,E_exact,pd,pd_accum,PD_exact,path_plot,t,False) 
        
            if t == epochs-1:
                exec_time = "{:2.0f}".format(time.time()-start_time)
                minimisation_plots(k,ann_s,ann_d,psi_exact_s_normalized,psi_exact_d_normalized,ks_accum,kd_accum,ks,kd,
                                   K_accum,K,U,E,U_accum,E_accum,E_exact,pd,pd_accum,PD_exact,path_plot,t,save_plot,exec_time)
                
        print('Model trained!')
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
        print(f'Skipping already trained model {name_of_file}...')
        
    filenumber += 1
    
print("\nAll done! :)")
   
