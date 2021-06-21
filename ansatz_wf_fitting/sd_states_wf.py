# -*- coding: utf-8 -*-

"""
This program trains an Artificial Neural Network to fit an ansatz 2-state wavefunction  
that we know is similar to the S and D states of the deuteron. The ANN has one input and 
two outputs, and uses a single hidden layer with Nhid hidden neurons. A 64-points tangential 
mesh is used as the momentum space upon which we train the ANN.

Side note: if the variable 'save' is set to True, the model will be automatically
stored in the path set in the 'path' variable at the beginning of the EPOCH LOOP. 
"""

import torch, time, integrator, math
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

############################## INITIAL STUFF #################################
device='cpu' 
print("Using {} device".format(device))
seed = 1
torch.manual_seed(seed) # Prevents the seed from changing at every run
save_model = False # if set to True, saves the model parameters and the plot
save_plot = False
show_pics = True
epochs = 25000

# General parameters
Nin = 1
Nout = 2
Nhid = 20
q_max = 500
n_samples = 64
n_test = 64
train_a = 0
train_b = q_max
test_a = 0
test_b = 5
q_max_quad = q_max

# Training/Test sets
x, w = np.polynomial.legendre.leggauss(n_samples)
a = torch.tensor(train_a)
b = torch.tensor(train_b)

x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # q mesh from 0 to 1
k = [(q_max/math.tan(x_i[-1]*math.pi/2))*math.tan(float(e)*math.pi/2) for e in x_i] # tangential q mesh from 0 to q_max
w_i = [torch.tensor(float(e)/2) for e in w] # GL weights
cos2 = [1/(math.cos(float(x_i[i])*math.pi/2))**2 for i in range(64)]
w_i = [w_i[r]*cos2[r] for r in range(64)]
p = (q_max/math.tan(x_i[-1]*math.pi/2))*math.pi/2 # multiplicative factor
Q_train = torch.tensor(k) 
q_2 = Q_train**2
Q_test = torch.linspace(test_a,test_b,n_test)

# ANN Parameters
W1 = torch.rand(Nhid,Nin,requires_grad=True)*(-1.) # First set of coefficients
B = torch.rand(Nhid)*2.-torch.tensor(1.) # Set of bias parameters
W2 = torch.rand(Nout,Nhid,requires_grad=True) # Second set of coefficients

# Target wavefunctions
psi_ansatz_s = torch.exp((-1.5**2)*Q_train**2/2) # target function L=0
psi_ansatz_s_test = torch.exp((-1.5**2)*Q_test**2/2)
psi_ansatz_d = (Q_train**2)*torch.exp((-1.5**2)*Q_train**2/2) # target function L=2
psi_ansatz_d_test = (Q_test**2)*torch.exp((-1.5**2)*Q_test**2/2)

def gl64(function):
    """Integrates a function given via tensor by Gauss-Legendre"""
    return p*torch.dot(torch.stack(w_i),function)

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")

def pic_creator(t,s):
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(wspace=0.25,hspace=0.35)
    
    # L=0
    plt.subplot(2,2,1)
    plt.title("S-state wave functions. Epoch {}".format(t+1),fontsize=15)
    plt.xlabel("$q\,(\mathrm{fm}^{-1})$",fontsize=15)
    plt.ylabel("$\psi\,(L=0)$",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot(np.linspace(0,5,n_test),[e.item() for e in psi_s_pred],label='$\psi_{\mathrm{ANN}}$')
    plt.plot(np.linspace(0,5,n_samples),psi_ansatz_s_normalized.tolist(),label='$\psi_{\mathrm{targ}}$')
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
    plt.plot(np.linspace(0,5,n_test),[e.item() for e in psi_d_pred],label='$\psi_{\mathrm{ANN}}$')
    plt.plot(np.linspace(0,5,n_samples),psi_ansatz_d_normalized.tolist(),label='$\psi_{\mathrm{targ}}$')
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
        plot_path = path + 'time{}.pdf'.format(exec_time)
        plt.savefig(plot_path)
        print('\nPicture saved in {}'.format(plot_path))
    plt.pause(0.001)

############################### THE ANN ######################################
# We create our nn class as a child class of nn.Module
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

# We load our psi_ann to the GPU (or CPU)
psi_ann = NeuralNetwork().to(device) 
print("NN architecture:\n\n",psi_ann)
show_layers(psi_ann)

############################ VARIOUS FUNCTIONS ###############################
norm_s1 = lambda q : (q**2)*(torch.exp((-1.5**2)*q**2))
norm_s = integrator.gl64(norm_s1,test_a,test_b)
psi_ansatz_s_normalized = psi_ansatz_s_test/torch.sqrt(norm_s)

norm_d1 = lambda q : (q**2)*((q**2)*torch.exp((-1.5**2)*q**2/2))**2
norm_d = integrator.gl64(norm_d1,test_a,test_b)
psi_ansatz_d_normalized = psi_ansatz_d_test/torch.sqrt(norm_d)

############################ COST FUNCTION ###################################
def cost():
    """Returns the cost"""
    # C(L=0)
    global ann_s,ann_d
    ann_s = []
    ann_d = []
    for i in range(n_samples):
        v = psi_ann(Q_train[i].unsqueeze(0))
        ann_s.append(v[0])
        ann_d.append(v[1])
    ann1 = torch.stack(ann_s)  
    ann2 = torch.stack(ann_d)
    
    global k_s
    num = ann1*psi_ansatz_s*q_2
    denom1 = q_2*(ann1**2) 
    nume = (gl64(num))**2
    den1 = gl64(denom1)
    den2 = norm_s
    k_s = nume/(den1*den2)
    c_s = (k_s-torch.tensor(1.))**2

    # C(L=2)
    global k_d
    num = ann2*psi_ansatz_d*q_2
    denom1 = q_2*(ann2**2) 
    nume = (gl64(num))**2
    den1 = gl64(denom1)
    den2 = norm_d
    k_d = nume/(den1*den2)
    c_d = (k_d-torch.tensor(1.))**2
    
    return c_s+c_d

######################### TRAINING AND TESTING ###############################
# Training parameters
learning_rate = 1e-2
epsilon = 1e-8
smoothing_constant = 0.9
loss_fn = cost
optimizer = torch.optim.RMSprop(params=psi_ann.parameters(),lr=learning_rate,eps=epsilon,alpha=smoothing_constant)

def train_loop(loss_fn,optimizer):
    """Trains the Neural Network over the training dataset"""    
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

psi_s_pred = []
psi_d_pred = []
def test_loop(test_set,model):
    """Generates a list with the predicted values"""
    with torch.no_grad():
        # Current wavefunction normalization 
        # Note: we use the slow integrator because it needs not be passed a specific mesh 
        norm_s2 = lambda q : (q**2)*(model(q)[0])**2
        norm_d2 = lambda q : (q**2)*(model(q)[1])**2
        norm_s_ann = integrator.gl64(norm_s2,test_a,test_b)
        norm_d_ann = integrator.gl64(norm_d2,test_a,test_b)
        
        for x in test_set:
            pred_s = model(x.unsqueeze(0).unsqueeze(0))[0]
            pred_d = model(x.unsqueeze(0).unsqueeze(0))[1]
            psi_s_pred.append(pred_s/torch.sqrt(norm_s_ann))
            psi_d_pred.append(pred_d/torch.sqrt(norm_d_ann))

################################ EPOCH LOOP ##################################
sep = "("
path = "../saved_models/s_d_states/nlayers1/fully_connected_ann/Sigmoid/lr0.01/nhid{}_epoch{}_seed{}_lr{}_actfun{}_optimizer{}_".format(Nhid,epochs,seed,learning_rate,
                                                                              str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                              str(optimizer).split(sep,1)[0].strip())
path = 'testt'
overlap_s = []
overlap_d = []
leap = 100 # number of iterations between each picture
start_time = time.time()

for t in range(epochs):        
    train_loop(loss_fn,optimizer)
    overlap_s.append(k_s.item())
    overlap_d.append(k_d.item())
    
    if ((t+1)%leap)==0:
        print(f"\nEpoch {t+1}\n-----------------------------------")
        if show_pics == True:
            test_loop(Q_test,psi_ann)
            pic_creator(t,False)
            psi_s_pred = []
            psi_d_pred = []
            
    if t == epochs-1:
        test_loop(Q_test,psi_ann)
        exec_time = "{:2.0f}".format(time.time()-start_time)
        pic_creator(t,save_plot)
        psi_s_pred = []
        psi_d_pred = []
        
full_path = '{}time{}.pt'.format(path,exec_time)

if save_model == True:
    torch.save(psi_ann.state_dict(),full_path)
    print('Model saved in {}'.format(full_path))

print("Execution time: {:6.2f} seconds (run on {})".format(float(exec_time),device))
print("\nDone!")

#show_layers(psi_ann) # uncomment to display the parameters of the trained ANN






    