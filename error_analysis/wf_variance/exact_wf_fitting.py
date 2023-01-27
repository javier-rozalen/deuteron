"""
We train a neural network to learn the 64 points of the exact deuteron states
so as to have an "exact" function that we can call at any point. Then we compute
the overlap between this network and the ones with overfitting to determine
the "real" fidelities. We do the same for the energies
"""
#%% IMPORTS, VARIABLES AND NEURAL NETWORK
# -*- coding: utf-8 -*-
import torch, time, math, os, re, N3LO, pathlib, statistics
from scipy import integrate
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

############################## DATA PREPARATION ##############################
"""For a particular 'nhid' set at the beginning, we load all the filtered initializations,
we compute the wave function and we store it in a the text file 'wf_nhid{}.txt'.

Parameters to adjust manually:
    device --> cpu, gpu
    save_plot --> Boolean
    test_a --> float
    test_b --> float
    n_test --> integer
    seed --> integer
    titles and other plotting stuff, if necessary
    
"""

device = 'cpu'
save_plot = False
test_a = 0
test_b = 4
n_test = 64
seed = 1
plot_q_train = False
output_model = 'exact_wf_fit_model.pt'
torch.manual_seed(seed)

# Gauss-Legendre 
x, w = np.polynomial.legendre.leggauss(n_test)
x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # q mesh from 0 to 1
w_i = [torch.tensor(float(e)/2) for e in w] # GL weights
cos2 = [1/(math.cos(float(x_i[i])*math.pi/2))**2 for i in range(n_test)]
p = (500/math.tan(x_i[-1]*math.pi/2))*math.pi/2 # multiplicative factor
w_i = torch.stack([p*w_i[r]*cos2[r] for r in range(n_test)])
k_gl_tan = [500/math.tan(x_i[-1]*math.pi/2)*math.tan(float(e)*math.pi/2) for e in x_i] 

# Uniform
k_uniform = torch.linspace(test_a,test_b,n_test)

# Gauss-Legendre without tangential transformation
k_gl_notan = [torch.tensor(float(e)*4) for e in x_i]

q_train = torch.tensor(k_gl_tan)

nlayers = 2
arch = 'fully_connected_ann'
initial_dir = pathlib.Path(__file__).parent.resolve()
path_to_data = f'{initial_dir}/{arch}/nlayers{nlayers}'
path_to_mimics = f'{path_to_data}/mimics'
list_of_files = []
path_to_output_files = '{}/nlayers{}/mimics/'.format(arch,nlayers)
epochs = 20000
leap = 500

# We fill a list with nhid of each of the good files
for file in os.listdir(path_to_data):
    if len(file.split('.')) > 1:
        list_of_files.append(file)

# We sort the list of pretrained models by nhid
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]
list_of_files.sort(key=num_sort)

def gl64(function):
    """Integrates a function given via tensor by Gauss-Legendre"""
    return torch.dot(w_i,function)

# We set the general parameters and the ANN
Nin = 1
Nout = 2

# Physical parameters
mu = 938.9182 # MeV
V_r = 1438.723 # MeV·fm
V_a = 626.893 # MeV·fm
hbar = 197.32968 # MeV·fm
m_r = 3.11 # fm^-1
m_a = 1.55 # fm^-1
pi = math.pi

def pic_creator():
    fig, ax = plt.subplots(nrows=1,ncols=2)
    fig.tight_layout()
    ax0 = ax[0]
    ax1 = ax[1]
    s = [e.detach().numpy() for e in pred_s]
    d = [e.detach().numpy() for e in pred_d]
    
    # S-state
    ax0.set_xlabel('q $(\mathrm{fm}^{-1})$')
    ax0.set_ylabel('psi (L=0)')
    ax0.set_xlim(0,2)
    ax0.plot(q_train,s,label='pred')
    ax0.plot(q_train,targ_s,label='targ')
    ax0.legend()
    
    # D-state
    ax1.set_xlabel('q $(\mathrm{fm}^{-1})$')
    ax1.set_ylabel('psi (L=2)')
    ax1.set_xlim(0,4)
    ax1.plot(q_train,d,label='pred')
    ax1.plot(q_train,targ_d,label='targ')
    ax1.legend()    
    
    fig.suptitle(f'Epoch {epoch+1}')
    
    plt.show()
    plt.pause(0.001)
    

############################### THE ANN ######################################
# We create our nn class as a child class of nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):       
        super(NeuralNetwork, self).__init__()
         
        # We set the operators 
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Softmax(dim=0) # activation function
        self.lc2 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        
        # We set the parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
   
    # We set the architecture
    def forward(self, x): 
        o = self.actfun(self.lc1(x))
        o = self.lc2(o)
        return o.squeeze()[0],o.squeeze()[1]

#%% TRAINING OF THE ANN
Nhid = 100

# Target wavefunction (Arnau/James)
q_2 = q_train**2
n3lo = N3LO.N3LO(f"{initial_dir}/../../deuteron_data/2d_vNN_J1.dat",
                 f"{initial_dir}/../../deuteron_data/wfk.dat")
psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)   
psi_target_s_norm = gl64(q_2*(psi_target_s)**2)
psi_target_d_norm = gl64(q_2*(psi_target_d)**2)
psi_target_s_normalized = psi_target_s/torch.sqrt(psi_target_s_norm+psi_target_d_norm)
psi_target_d_normalized = psi_target_d/torch.sqrt(psi_target_d_norm+psi_target_s_norm)
targ_s = psi_target_s_normalized
targ_d = psi_target_d_normalized

# ANN Parameters
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) # First set of coefficients
B = torch.rand(Nhid)*2.-torch.tensor(1.) # Set of bias parameters
W2 = torch.rand(Nout,Nhid,requires_grad=True) # Second set of coefficients

psi_ann = NeuralNetwork().to(device)        

def cost(targ_s,targ_d):
    """MSE summed over all states"""
    ds = 0
    dd = 0
    global pred_s,pred_d
    pred_s = []
    pred_d = []

    for q in q_train:
        psi_s,psi_d = psi_ann(q.unsqueeze(0))
        pred_s.append(psi_s)
        pred_d.append(psi_d)
    for i in range(len(pred_s)):
        ds += (pred_s[i]-targ_s[i])**2
        dd += 1000*(pred_d[i]-targ_d[i])**2
    return ds+dd

# ANN Hyperparameters
learning_rate = 1e-1
epsilon = 1e-8
smoothing_constant = 0.9
loss_fn = cost
optimizer = torch.optim.RMSprop(params=psi_ann.parameters(), lr=learning_rate,
                                eps=epsilon, alpha=smoothing_constant)
optimizer = torch.optim.Adam(params=psi_ann.parameters(), lr=learning_rate,
                             eps=epsilon)
#optimizer = torch.optim.SGD(params=psi_ann.parameters(),lr=learning_rate)

# Training loop
def train_loop():
    """Trains the Neural Network over the training dataset"""    
    optimizer.zero_grad()
    loss_fn(targ_s, targ_d).backward()
    optimizer.step()
    
# Epoch loop
for epoch in range(epochs):
    train_loop()
    if ((epoch+1)%leap) == 0:
        pic_creator()
    
torch.save(psi_ann.state_dict(), output_model)
 
   
#%% TESTING THE NEW NETWORK AGAINST THE OLD ONE
q_test = torch.linspace(0, 4, 1000)
############# RESULTS #############
psi_ann.load_state_dict(torch.load(output_model))
psi_ann.eval()
pred_s = []
pred_d = []
for q in q_test:
    psi_s,psi_d = psi_ann(q.unsqueeze(0))
    pred_s.append(psi_s)
    pred_d.append(psi_d)
s = [e.detach().numpy() for e in pred_s]
d = [e.detach().numpy() for e in pred_d]


############# PLOTTING ##################
fig, ax = plt.subplots(nrows=1,ncols=2)
fig.tight_layout()
plt.subplots_adjust(top=0.9)
ax0 = ax[0]
ax1 = ax[1]

# S-state
ax0.set_xlabel('q $(\mathrm{fm}^{-1})$')
ax0.set_ylabel('psi (L=0)')
ax0.set_xlim(0,2)
ax0.set_ylim(-0.5,14)
ax0.plot(q_test,s,label='ANN fit')
ax0.plot(q_train,targ_s,label='minim. ANN')
#ax0.plot(q_test,interpol_s,label='interpol')
ax0.legend()

# D-state
ax1.set_xlabel('q $(\mathrm{fm}^{-1})$')
ax1.set_ylabel('psi (L=2)')
ax1.set_xlim(0,4)
ax1.plot(q_test,d,label='ANN fit')
ax1.plot(q_train,targ_d,label='minim. ANN')
for q in q_train:
    ax1.axvline(q,linewidth=0.5,linestyle='--')
#ax1.plot(q_test,interpol_d,label='interpol')
ax1.legend()    

fig.suptitle(f'Epoch {epoch+1}')

plt.show()
plt.pause(0.001)

    
#%% COMPUTING THE OVERLAP BETWEEN THE NEW NETWORK AND THE OVERFITTED ONE. 
# ALSO THE KINETIC ENERGY
"""
Note 1: before running this cell, make sure to generate the overfitted functions
on a linear mesh with enough points so that the integral can properly capture
the steps. Make sure that the variables n_test,q_test correspond to this mesh!!

Note 2: we use the standard Simpson integration method.
"""            
n_test = 500
q_test = torch.linspace(0,10,n_test) 
q_test_2 = q_test**2

fidelities_s = list()
fidelities_d = list()
kinetic_energies = list()
for nhid in [20,30,40,60,80,100]:
    print(f'\nNhid = {nhid}')
    for instance_number in range(0,18):
        #### DATA PREPARATION ####
        # Overfitted function
        dictionary_of_seeds = dict()
        model = f'wf_variance_nhid{nhid}_ntest{n_test}.txt'
        #print(f'Using model {model}')
        ov_s = [] # overfitted state S
        ov_d = [] # overfitted state D
        with open(f'{path_to_data}/{model}','r') as f:
            lines = f.readlines()
            c = 0
            for line in lines:
                if len(line) > 1:
                    ov_s.append(float(line.split(' ')[0]))
                    ov_d.append(float(line.split(' ')[1]))
                else:
                    dictionary_of_seeds[c] = ov_s + ov_d
                    ov_s = []
                    ov_d = []
                    c += 1
                              
        # 'Interpolation' function    
        W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) # First set of coefficients
        B = torch.rand(Nhid)*2.-torch.tensor(1.) # Set of bias parameters
        W2 = torch.rand(Nout,Nhid,requires_grad=True) # Second set of coefficients
        
        psi_ann2 = NeuralNetwork().to(device)
        psi_ann2.load_state_dict(torch.load('model.pt'))
        psi_ann2.eval()
        
        # We store the predictions of this net in two lists: s,d
        interpol_s = []
        interpol_d = []
        for q in q_test:
            psi_s,psi_d = psi_ann2(q.unsqueeze(0))
            interpol_s.append(psi_s)
            interpol_d.append(psi_d)
        s = [e.detach().numpy() for e in interpol_s]
        d = [e.detach().numpy() for e in interpol_d]
        
        ######## INTEGRATION #########
        """
        # Testing simpson
        x = np.linspace(0,3,100)
        y = [np.sin(k) for k in x]
        I_test = integrate.simpson(y,x)"""
        
        # Norm of ANN fit S
        # Note: this function should already have norm approx. to 1 since it is
        #       the fitting to a normalized function
        squared_s = torch.tensor([i**2 for i in s])*q_test_2
        squared_d = torch.tensor([i**2 for i in d])*q_test_2
        norm_ann_fit_s = integrate.simpson(squared_s,q_test)
        norm_ann_fit_d = integrate.simpson(squared_d,q_test)
        s_normalized = [x/np.sqrt(norm_ann_fit_s+norm_ann_fit_d) for x in s]
        d_normalized = [x/np.sqrt(norm_ann_fit_s+norm_ann_fit_d) for x in d]
        
        # Norm of minim. ANN
        # Note: this function should already have norm approx. to 1 since it 
        #       comes from a minimisation in which we normalize at every step
        squared_s = torch.tensor([i**2 for i in dictionary_of_seeds[instance_number][:n_test]])*q_test_2
        squared_d = torch.tensor([i**2 for i in dictionary_of_seeds[instance_number][n_test:]])*q_test_2
        norm_ann_minim_s = integrate.simpson(squared_s,q_test)
        norm_ann_minim_d = integrate.simpson(squared_d,q_test)
        minim_ann_s_normalized = [x/np.sqrt(norm_ann_minim_s+norm_ann_minim_d) for x in dictionary_of_seeds[instance_number][:n_test]]
        minim_ann_d_normalized = [x/np.sqrt(norm_ann_minim_s+norm_ann_minim_d) for x in dictionary_of_seeds[instance_number][n_test:]]
        minim_ann_s = [x for x in dictionary_of_seeds[instance_number][:n_test]]
        minim_ann_d = [x for x in dictionary_of_seeds[instance_number][n_test:]]
        
        # OVERLAP
        integrable_s = [i*j*k for i,j,k in zip(s,minim_ann_s,q_test_2)]
        integrable_d = [i*j*k for i,j,k in zip(d,minim_ann_d,q_test_2)]     
        I_s = integrate.simpson(integrable_s,q_test)**2/(norm_ann_minim_s*norm_ann_fit_s)   
        I_d = integrate.simpson(integrable_d,q_test)**2/(norm_ann_minim_d*norm_ann_fit_d)
        #I = I_s + I_d    
        
        # KINETIC ENERGY
        minim_ann_s_plot = [x*np.sqrt(norm_ann_fit_s) for x in dictionary_of_seeds[instance_number][:n_test]]
        minim_ann_d_plot = [x*np.sqrt(norm_ann_fit_d) for x in dictionary_of_seeds[instance_number][n_test:]]
        minim_ann_s_normalized_tensor = [torch.tensor(e) for e in minim_ann_s_plot]
        minim_ann_d_normalized_tensor = [torch.tensor(e) for e in minim_ann_d_plot]
        minim_ann_sd = torch.cat((torch.stack(minim_ann_s_normalized_tensor),torch.stack(minim_ann_d_normalized_tensor)))
        q_test_4_sd = torch.cat((q_test_2**2,q_test_2**2))
        k_integrand = q_test_4_sd*minim_ann_sd**2
        K = (hbar**2/mu)*integrate.simpson(k_integrand,torch.cat((q_test,q_test)))  
        
        print(f'\nInstance {instance_number}')
        print(f'Fidelity S = {abs(I_s)}')
        print(f'Fidelity D = {abs(I_d)}')
        print(f'Kinetic energy = {K} MeV')
        fidelities_s.append(abs(I_s)) 
        fidelities_d.append(abs(I_d)) 
        kinetic_energies.append(K)    
       
        
        ############# PLOTTING ##################
        
        fig, ax = plt.subplots(nrows=1,ncols=2)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.4,top=0.9)
        ax0 = ax[0]
        ax1 = ax[1]
        
        # S-state
        ax0.set_xlabel('q $(\mathrm{fm}^{-1})$')
        ax0.set_ylabel('psi (L=0)')
        ax0.set_xlim(0,2)
        ax0.set_ylim(-14,14)
        ax0.plot(q_test,s,label='ANN fit')
        ax0.plot(q_test,minim_ann_s_plot,label='minim. ANN')
        ax0.legend()
        
        # D-state
        ax1.set_xlabel('q $(\mathrm{fm}^{-1})$')
        ax1.set_ylabel('psi (L=2)')
        ax1.set_xlim(0,4)
        ax1.set_ylim(-.25,0.25)
        ax1.plot(q_test,d,label='ANN fit')
        ax1.plot(q_test,minim_ann_d_plot,label='minim. ANN')
        ax1.legend()    
        
        fig.suptitle(f'Nhid = {nhid}, Instance = {instance_number}')
        plt.show()
        plt.pause(0.001) 
            
fidelities_s.sort()
fidelities_d.sort()
kinetic_error = [abs(14.638690-e) for e in kinetic_energies]
kinetic_error.sort()
print(f'\nMinimum F^S: {min(fidelities_s)}')
print(f'Minimum F^D: {min(fidelities_d)}')
print(f'Maximum Kinetic energy absolute error: {max(kinetic_error)} MeV')
print(f'Maximum Kinetic energy relative error: {100*max(kinetic_error)/14.638690} %')
print(f'Stdev of K = {statistics.stdev(kinetic_energies)}')



















