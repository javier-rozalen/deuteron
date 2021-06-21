# -*- coding: utf-8 -*-

"""
This program determines the error in the overlap, total energy and probability
of D-state given a fully-trained model. 

Method: let the system evolve for 100 epochs approx., store the parameters and 
find the maximum, minimum and the average. 
"""

import torch, time, math, N3LO
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor

############################## INITIAL STUFF #################################
device='cpu' 
seed = 8
torch.manual_seed(seed) # Prevents the seed from changing at every run
save_model = False # if set to True, saves the model parameters
show_pics = True
save_plot = False
save_data = False
epochs = 300

# General parameters
Nin = 1
Nout = 2
q_max = 500
n_samples = 64
train_a = 0
train_b = q_max
# Multi-layer parameters
Hd = 4
nlayers = 2
# Test parameters (only if test_loop function is not empty)
n_test = 64
test_a = 0
test_b = 5

# Single-layer specific params
Nhid = 8
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) # First set of coefficients
B = torch.rand(Nhid)*2.-torch.tensor(1.) # Set of bias parameters
W2 = torch.rand(Nout,Nhid,requires_grad=True) # Second set of coefficients

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
w_i = [p*w_i[r]*cos2[r] for r in range(64)]
w_i = torch.stack(w_i)
Q_train = torch.tensor(k) 
q_2 = Q_train**2
Q_test = torch.linspace(test_a,test_b,n_test)

# Multi-layer ANN layer architecture
Layers = [Nin,Hd,Hd,Nout]

def gl64(function):
    """Integrates a function given via tensor by Gauss-Legendre"""
    return torch.dot(w_i,function)

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")

def pic_creator(t,s):
    
    plt.figure(figsize=(18,8))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
        
    # Overlap
    plt.subplot(1,3,1)
    plt.title("Overlap")
    plt.xlabel("Epoch")
    plt.ylabel("Fidelity")
    plt.ylim(0.9998,1.0)
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
    
    
    #plt.legend(fontsize=12)
           
    # E
    plt.subplot(1,3,2)
    plt.title("Total Energy")
    plt.xlabel("Epoch")
    plt.ylabel("E (MeV)")
    #plt.ylim(-2.227,-2.216)
    plt.ylim(-2.227,-2.221)
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
    plt.ylim(4.3,4.7)
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

    if s == True:
        plot_path = '{}time{}.png'.format(path,exec_time)
        plt.savefig(plot_path)
        print('\nPicture saved in {}'.format(plot_path))
    plt.pause(0.001)
  

############################### SIGNLE-LAYER ANN ######################################
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
    
"""    
############################### N-LAYER ANN ######################################
# We create our nn class as a child class of nn.Module
class NeuralNetwork(nn.Module):
    #The ANN takes the layer configuration as an input via a list (Layers)
    
    # Constructor
    def __init__(self, Layers):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.ModuleList()
        self.actfun = nn.Sigmoid()
        c = 0
        for input_size, output_size in zip(Layers, Layers[1:]):
            if c == 0:
                self.hidden.append(nn.Linear(input_size, output_size, bias=True))
            else:
                self.hidden.append(nn.Linear(input_size, output_size, bias=False))
            c += 1
            
        # We configure the parameters 
        #with torch.no_grad():
         #   self.hidden[0].bias = nn.Parameter(B)
   
    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = self.actfun(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation
""" 

# We load our psi_ann to the CPU (or GPU)
psi_ann = NeuralNetwork().to(device) # Single-layer ANN
#psi_ann = NeuralNetwork(Layers).to(device) # Extended ANN
psi_ann.load_state_dict(torch.load("../saved_models/n3lo/nlayers1/fully_connected_ann/Sigmoid/lr0.01/nhid8_epoch250000_seed8_lr0.01_actfunSigmoid_optimizerRMSprop_time3446.pt"))
psi_ann.eval()
print("NN architecture:\n",psi_ann)
show_layers(psi_ann)

############################ VARIOUS FUNCTIONS ###############################
# N3LO potential
n3lo = N3LO.N3LO("../deuteron_data/2d_vNN_J1.dat","../deuteron_data/wfk.dat")
V_ij = n3lo.getPotential()

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
optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,eps=epsilon,
                                alpha=smoothing_constant)

def train_loop(loss_fn,optimizer):
    """Trains the Neural Network over the training dataset"""    
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

#psi_s_pred = []
#psi_d_pred = []
def test_loop(test_set,model):
    """Generates a list with the predicted values"""
    with torch.no_grad():
        # Current wavefunction normalization 
        # Note: we use the slow integrator because it needs not be passed a specific mesh 
        pass

################################ EPOCH LOOP ##################################
sep = "("
path = "saved_models/n3lo/nlayers2/Sigmoid/lr0.001/Hd{}_nlayers{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}_".format(Hd,nlayers,epochs,seed,learning_rate,
                                                                              str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                              str(optimizer).split(sep,1)[0].strip())

# We store the energy data in lists so as to plot it
K_accum=[]
U_accum=[]
E_accum=[]
ks_accum=[]
kd_accum=[]
pd_accum=[]
leap = 50

def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

# Error analysis parameters
mean_E_accum=[]
E_top_accum=[]
E_bot_accum=[]
mean_ks_accum=[]
mean_kd_accum=[]
ks_top_accum=[]
ks_bot_accum=[]
kd_top_accum=[]
kd_bot_accum=[]
mean_PD_accum=[]
pd_top_accum=[]
pd_bot_accum=[]
mean_E = 0
mean_E_top = 0
mean_E_bot = 0
mean_ks = 0
mean_kd = 0
mean_ks_top = 0
mean_ks_bot = 0
mean_kd_top = 0
mean_kd_bot = 0
mean_PD = 0
mean_pd_top = 0
mean_pd_bot = 0

try: 
    def f():
        min_take = 120
        
        global mean_E_top,mean_E_bot,mean_ks_top,mean_ks_bot,mean_kd_top,mean_kd_bot
        global mean_pd_top,mean_pd_bot
        mean_E_top = max(E_accum[min_take:])
        mean_E_bot = min(E_accum[min_take:])
        
        mean_ks_top = max(ks_accum[min_take:])
        mean_ks_bot = min(ks_accum[min_take:])
        
        mean_kd_top = max(kd_accum[min_take:])
        mean_kd_bot = min(kd_accum[min_take:])
        
        mean_pd_top = max(pd_accum[min_take:])
        mean_pd_bot = min(pd_accum[min_take:])
        
        pic_creator(t,False)
        
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
        
        pic_creator(t,save_plot)
        
    start_time = time.time()
    for t in range(epochs):        
        train_loop(loss_fn,optimizer)
        K_accum.append(K.item())
        U_accum.append(U.item())
        E_accum.append(E.item())
        ks_accum.append(ks.item())
        kd_accum.append(kd.item())
        pd_accum.append(pd.item())
        
        if t>20: 
            
            ########################### ERROR ################################
            # Energy
            mean_E = sum(E_accum[20:])/len(E_accum[20:])
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
            
            # Overlap S
            mean_ks = sum(ks_accum[20:])/len(ks_accum[20:])
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
            mean_kd = sum(kd_accum[20:])/len(kd_accum[20:])
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
            mean_PD = sum(pd_accum[20:])/len(pd_accum[20:])
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
            
        
        if ((t+1)%leap)==0:
            print(f"\nEpoch {t+1}\n-----------------------------------")
            if show_pics == True:
                # Uncomment the following lines if test_loop function is not empty
                #test_loop(Q_test,psi_ann)
                pic_creator(t,False) 
                #psi_s_pred = []
                #psi_d_pred = []
    
        if t == epochs-1:
            test_loop(Q_test,psi_ann)
            exec_time = "{:2.0f}".format(time.time()-start_time)
            pic_creator(t,save_plot)
            psi_s_pred = []
            psi_d_pred = []
            
    f()

except KeyboardInterrupt:
    exec_time = "{:2.0f}".format(time.time()-start_time)
    f()

full_path = '{}time{}.pt'.format(path,exec_time)

if save_data == True:
    filename= 'nlayers1/nhid{}.txt'.format(Nhid) # single-layer ANN
    #filename= 'nlayers2/Hd{}.txt'.format(Hd) # double-layer ANN
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
                    str(round_to_1(abs(abs(mean_pd_bot)-abs(mean_PD))))+' \n')
    file.close()
    print(f"\nData saved in {filename}.")
    
if save_model == True:
    torch.save(psi_ann.state_dict(),full_path)
    print("Model saved in {}_time{}.pt".format(path,exec_time))
        
print("Execution time: {:6.2f} seconds (run on {})".format(time.time()-start_time,device))
print("\nDone!")

#show_layers(psi_ann)




    