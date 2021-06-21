# -*- coding: utf-8 -*-

"""
This program trains an Artificial Neural Network to find the deuteron wave function. The ANN  
has one input and two outputs, and uses a single hidden layer with Nhid hidden neurons. A  
64-points tangential mesh is used as the momentum space upon which we train the ANN. 
The potential used is N3LO.

Side note: if the variable 'save' is set to True, the model will be automatically 
stored in the path set in the 'path' variable at the beginning of the EPOCH LOOP. 
"""

import torch, time, math, N3LO
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

############################## INITIAL STUFF #################################
device='cpu' 
seed = 1
torch.manual_seed(seed) # Prevents the seed from changing at every run
save_model = False 
save_plot = False
show_pics = True
epochs = 500

# General parameters
Nin = 1
Nout = 2
Nhid = 8
q_max = 500
n_samples = 64
train_a = 0
train_b = q_max
# Test parameters (only if test_loop function is not empty)
n_test = 64 
test_a = 0
test_b = 5

# Physical parameters
mu = 938.9182 # MeV
V_r = 1438.723 # MeV·fm
V_a = 626.893 # MeV·fm
hbar = 197.32968 # MeV·fm
m_r = 3.11 # fm^-1
m_a = 1.55 # fm^-1
pi = math.pi

# Training/Test sets
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # q mesh from 0 to 1
# tangential q mesh from 0 to q_max
k = [(q_max/math.tan(x_i[-1]*math.pi/2))*math.tan(float(e)*math.pi/2) for e in x_i] 
w_i = [torch.tensor(float(e)/2) for e in w] # GL weights
cos2 = [1/(math.cos(float(x_i[i])*math.pi/2))**2 for i in range(64)]
p = (q_max/math.tan(x_i[-1]*math.pi/2))*math.pi/2 # multiplicative factor
w_i = torch.stack([p*w_i[r]*cos2[r] for r in range(64)])
Q_train = torch.tensor(k) 
q_2 = Q_train**2
Q_test = torch.linspace(test_a,test_b,n_test) # test set (only if test_loop is not empty)

# ANN Parameters
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) # First set of coefficients
B = torch.rand(Nhid)*2.-torch.tensor(1.) # Set of bias parameters
W2 = torch.rand(Nout,Nhid,requires_grad=True) # Second set of coefficients

def gl64(function):
    """Integrates a function given via tensor by Gauss-Legendre"""
    return torch.dot(w_i,function)

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")

def pic_creator(t,s):
    plt.figure(figsize=(18,11))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    
    # L=0
    plt.subplot(2,4,1)
    plt.title("S-state wave functions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$")
    plt.ylabel("$\psi\,(L=0)$")
    plt.plot(k[:53],ann_s.detach()[:53],label='$\psi_{ANN}^{L=0}$')
    plt.plot(k[:53],psi_exact_s_normalized.numpy()[:53],label='$\psi_{\mathrm{targ}}^{L=0}$')
    plt.legend(fontsize=17)
    
    # L=2
    plt.subplot(2,4,2)
    plt.title("D-state wave functions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$")
    plt.ylabel("$\psi\,(L=2)$")
    plt.plot(k[:53],ann_d.detach()[:53],label='$\psi_{\mathrm{ANN}}^{L=2}$')
    plt.plot(k[:53],psi_exact_d_normalized.numpy()[:53],label='$\psi_{\mathrm{targ}}^{L=2}$')
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
        plot_path = path + 'time{}.png'.format(exec_time)
        plt.savefig(plot_path)
        print('\nPicture saved in {}'.format(plot_path))
    plt.pause(0.001)

############################### THE ANN ######################################
# We create our nn class as a child class of nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):       
        super(NeuralNetwork, self).__init__()
         
        # We set the operators 
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Sigmoid() # activation function
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

# We load our psi_ann to the CPU (or GPU)
psi_ann = NeuralNetwork().to(device) 
# To load a pre-trained model, uncomment the following lines and select the path of the model
#psi_ann.load_state_dict(torch.load("saved_models/s_d_states/nlayers1/fully_connected_ann/Sigmoid/lr0.01/nhid8_epoch250000_seed8_lr0.01_actfunSigmoid_optimizerRMSprop_time3446.pt"))
#psi_ann.eval()
print("NN architecture:\n\n",psi_ann)
show_layers(psi_ann)

############################ VARIOUS FUNCTIONS ###############################
# N3LO potential
n3lo = N3LO.N3LO("deuteron_data/2d_vNN_J1.dat","deuteron_data/wfk.dat")
V_ij = n3lo.getPotential()

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
psi_exact_s_norm = gl64(q_2*(psi_exact_s)**2)
psi_exact_d_norm = gl64(q_2*(psi_exact_d)**2)
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

############################ COST FUNCTION ###################################
def cost():
    """Returns the cost computed as the expected energy using the current wavefunction (ANN)
    Also returns the overlap with the theoretical wavefunction"""
    # Wavefunction 
    global ann_s,ann_d,norm2,K,U,E,ks,kd,pd
    ann_s1 = []
    ann_d1 = []
    for i in range(n_samples):
        v = psi_ann(Q_train[i].unsqueeze(0))
        ann_s1.append(v[0])
        ann_d1.append(v[1])
    ann1 = torch.stack(ann_s1)  
    ann2 = torch.stack(ann_d1)
    
    # Norm 
    norm_s = gl64(q_2*(ann1)**2)
    norm_d = gl64(q_2*(ann2)**2)    
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
    ks = ((gl64(q_2*ann1*psi_exact_s))**2)/(norm_s*psi_exact_s_norm)
    
    # Overlap (L=2)
    kd = ((gl64(q_2*ann2*psi_exact_d))**2)/(norm_d*psi_exact_d_norm)
    
    return E

######################### TRAINING AND TESTING ###############################
# Training parameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9
loss_fn = cost
optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,
                                eps=epsilon,alpha=smoothing_constant)

def train_loop(loss_fn,optimizer):
    """Trains the Neural Network on the training dataset"""    
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

psi_s_pred = []
psi_s_pred = []
def test_loop(test_set,model):
    """Generates a list with the predicted values. This can be used to test the
       wave function outside the training set."""
    with torch.no_grad():
        pass

################################ EPOCH LOOP ##################################
sep = "("
path = "saved_models/n3lo/nlayers1/fully_connected_ann/Sigmoid/lr0.01/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}_".format(Nhid,epochs,seed,learning_rate,
                                                                              str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                              str(optimizer).split(sep,1)[0].strip())

# We store the energy data in lists so as to plot it
K_accum=[]
U_accum=[]
E_accum=[]
ks_accum=[]
kd_accum=[]
pd_accum=[]
leap = 100 # numer of iterations between each picture

start_time = time.time()
for t in range(epochs):        
    train_loop(loss_fn,optimizer)
    K_accum.append(K.item())
    U_accum.append(U.item())
    E_accum.append(E.item())
    ks_accum.append(ks.item())
    kd_accum.append(kd.item())
    pd_accum.append(pd.item())

    if ((t+1)%leap)==0:
        print(f"\nEpoch {t+1}\n-----------------------------------")
        if show_pics == True:
            # Uncomment the following lines if test_loop function is not empty
            #test_loop(Q_test,psi_ann)
            pic_creator(t,False) 
            #psi_s_pred = []
            #psi_d_pred = []

    if t == epochs-1:
        #test_loop(Q_test,psi_ann)
        exec_time = "{:2.0f}".format(time.time()-start_time)
        pic_creator(t,save_plot)
        psi_s_pred = []
        psi_d_pred = []
        
full_path = '{}time{}.pt'.format(path,exec_time)
full_path_image = '{}time{}.pdf'.format(path,exec_time)

if save_model == True:
    torch.save(psi_ann.state_dict(),full_path)
    print("Model saved in {}".format(full_path))

print("Execution time: {:6.2f} seconds (run on {})".format(time.time()-start_time,device))
print("\nDone!")

#show_layers(psi_ann) # uncomment to display the parameters of the trained ANN

    
