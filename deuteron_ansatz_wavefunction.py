# -*- coding: utf-8 -*-
"""
This program trains an Artificial Neural Network (ANN) to compute a physical wavefunction: the two bound states
of the Deuteron. Starting from a 'blank' ANN with 4Nhid parameters and a single hidden layer, we use it as the trial 
wavefunction in a Rayleigh-Ritz minimisation scheme, with the parameters being the 4Nhid ANN parameters. This 
program is the first step towards the energy minimisation: we train the ANN to take the form of an ansatz physical 
wavefunction to start with, and we do so by maximising the overlap of the ANN and the ansatz function. 

Note: if you do not wish to store the image and the model in a file, set the 'save' parameter to 'False'
"""
import torch, time, integrator, math
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

############################## INITIALIZATION VARIABLES #################################
device='cpu' # stores the ANN in the CPU 
seed = 1
torch.manual_seed(seed) # Prevents the seed from changing at every run
save = True # if set to True, saves the model parameters and the plot
epochs = 10000

# General parameters
Nin = 1
Nout = 2
Nhid = 10
q_max = 500
n_samples = 64
n_test = 64
train_a = 0
train_b = q_max
test_a = 0
test_b = 5
q_max_quad = q_max

# Training/Test sets
x, w = np.polynomial.legendre.leggauss(n_samples) # Gauss-Legendre with N=64 points
a = torch.tensor(train_a)
b = torch.tensor(train_b)

x_i = [torch.tensor(float(e)*0.5+0.5) for e in x] # momenta mesh from 0 to 1
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
    """Integrates a function evaluated over the mesh described above given via tensor by Gauss-Legendre"""
    return p*torch.dot(torch.stack(w_i),function)

def show_layers(model):
    """Prints the most important parameters of the Neural Network in the console"""
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")

def pic_creator(t,s):
    """Creates and plots an image of the wavefunction and its overlap with the given ansatz function.
    To save the image, pass the input parameter 's' with the value 'True'"""
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    
    # L=0
    plt.subplot(2,2,1)
    plt.title("S-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=0)$")
    plt.plot(np.linspace(0,5,n_test),psi_s_pred,label='$\psi_{ANN}$')
    plt.plot(torch.linspace(0,5,n_samples).numpy(),psi_ansatz_s_normalized.numpy(),label='$\psi_{targ}$')
    plt.legend()
    
    # Overlap (L=0)
    plt.subplot(2,2,2)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_s[-1]))
    plt.xlabel("Epoch")
    plt.ylabel("Overlap$\,(L=0)$")
    plt.plot(np.linspace(0,len(overlap_s),len(overlap_s)),overlap_s,label='$F^{S}$')
    plt.legend()
    
    # L=2
    plt.subplot(2,2,3)
    plt.title("D-state wavefunctions. Epoch {}".format(t+1))
    plt.xlabel("$q\,(fm^{-1})$")
    plt.ylabel("$\psi\,(L=2)$")
    plt.plot(np.linspace(0,5,n_test),np.array(psi_d_pred),label='$\psi_{ANN}$')
    plt.plot(np.linspace(0,5,n_samples),psi_ansatz_d_normalized.numpy(),label='$\psi_{targ}$')
    plt.legend()
    
    # Overlap (L=2)
    plt.subplot(2,2,4)
    plt.title("Overlap. Current fidelity: {:6.6f}".format(overlap_d[-1]))
    plt.xlabel("Epoch")
    plt.ylabel("Overlap$\,(L=2)$")
    plt.plot(torch.linspace(0,len(overlap_d),len(overlap_d)).numpy(),overlap_d,label='$F^{D}$')
    plt.legend()

    if s == True:
        plt.savefig(path+"time{}.png".format(exec_time))
    plt.pause(0.001)

############################### THE ANN ######################################
# We create our nn class as a child class of nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # We set the layer architecture 
        self.lc1 = nn.Linear(Nin,Nhid,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Sigmoid() # activation function
        self.lc2 = nn.Linear(Nhid,Nout,bias=False) # shape = (Nout,Nhid)
        
        # We set the initial parameters 
        with torch.no_grad():
            self.lc1.weight = nn.Parameter(W1)
            self.lc1.bias = nn.Parameter(B)
            self.lc2.weight = nn.Parameter(W2)
   
    # We set the feedforward operations
    def forward(self, x): 
        o = self.lc2(self.actfun(self.lc1(x)))
        return o.squeeze()[0],o.squeeze()[1]

# We load our psi_ann to the GPU (or CPU)
psi_ann = NeuralNetwork().to(device) 
print("NN architecture:\n",psi_ann)

############################ VARIOUS FUNCTIONS ###############################
# Normalization of the S-state ansatz wavefunction
norm_s1 = lambda q : (q**2)*(torch.exp((-1.5**2)*q**2))
norm_s = integrator.gl64(norm_s1,test_a,test_b)
psi_ansatz_s_normalized = psi_ansatz_s_test/torch.sqrt(norm_s)

# Normalization of the D-state ansatz wavefunction
norm_d1 = lambda q : (q**2)*((q**2)*torch.exp((-1.5**2)*q**2/2))**2
norm_d = integrator.gl64(norm_d1,test_a,test_b)
psi_ansatz_d_normalized = psi_ansatz_d_test/torch.sqrt(norm_d)

############################ COST FUNCTION ###################################
def cost():
    """Returns the cost that is to be minimised via backpropagation.
    Cost = (K^S-1)^2+(K^D-1)^2, where K^L is the overlap of the L-state."""
    
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
    """Trains the Neural Network"""    
    optimizer.zero_grad()
    loss_fn().backward()
    optimizer.step()

# We store the function values over the mesh at each epoch
psi_s_pred = []
psi_d_pred = []
def test_loop(test_set,model):
    """Generates a list with the predicted values"""
    with torch.no_grad():
        # Current wavefunction normalization 
        # Note: we use our external integrator because it needs not be passed a specific mesh 
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
# We store the model in the desired route, the name of the file containing relevant
# information about the run parameters
sep = "("
path = "models/s_d_states/epoch{}_seed{}_lr{}_actfun{}_optimizer{}_".format(epochs,seed,learning_rate,
                                                                              str(psi_ann.actfun).split(sep,1)[0].strip(),
                                                                              str(optimizer).split(sep,1)[0].strip())

# We store the overlap data in lists so as to plot them, similar to the psi_l_pred lists
overlap_s=[]
overlap_d=[]
start_time = time.time()
p_epochs = 500

# Epoch loop: we run the training loop a total of 'epochs' times
for t in range(epochs):        
    train_loop(loss_fn,optimizer) # Training
    overlap_s.append(k_s.item()) 
    overlap_d.append(k_d.item())
    
    # We plot the current results every p_epochs
    if ((t+1)%p_epochs)==0:
        print(f"\nEpoch {t+1}\n-----------------------------------")
        test_loop(Q_test,psi_ann)
        pic_creator(t,False) 
        psi_s_pred = []
        psi_d_pred = []
        
    # We plot and save (if set to True) the model at the last step
    if t == epochs-1:
        test_loop(Q_test,psi_ann)
        exec_time = "{:2.0f}".format(time.time()-start_time)
        pic_creator(t,save)
        psi_s_pred = []
        psi_d_pred = []
        
show_layers(psi_ann)
print("\nDone!")
print("Execution time: {:6.2f} seconds (run on {})".format(time.time()-start_time,device))

# We save the model, if 'save' is set to 'True'
if save == True:
    torch.save(psi_ann.state_dict(),path+"time{}.pt".format(exec_time))
    print("Model saved in {}_time{}.pt".format(path,exec_time))

############################ COMING SOON... ###################################
"""
# Malfliet-Tjon potential
V = [[0] * 64 for i in range(64)]
V_MT = lambda q,q1 : (2./(5.*pi**2))*(4/(q*q1))*(V_r*math.log(((q1+q)**2+m_r**2)/((q1-q)**2+m_r**2))-
                                            V_a*math.log(((q1+q)**2+m_a**2)/((q1-q)**2+m_a**2)))

for i in range(64):
    for j in range(64):
        V[i][j]=V_MT(Q_train[i],Q_train[j])
y = []

############################ COST FUNCTION ###################################
def cost():
    #Returns the cost
    global ann
    ann = [psi_ann(Q_train[i].unsqueeze(0)).squeeze(0) for i in range(n_samples)]
    ann1 = torch.stack(ann)
    ann1 = torch.stack(psi_target)
    
    # K (L=0)
    global K
    num = (1/mu)*(ann1*q_2)**2
    norm_ann1 = q_2*(ann1**2) 
    nume = gl64(num)
    norm_ann = gl64(norm_ann1)
    K = (hbar**2)*nume/norm_ann
    
    # U (L=0)
    global U
    y = q_2*torch.stack(w_i)*p*ann1
    V_ij = torch.tensor(V)
    U = torch.matmul(y,torch.matmul(V_ij,y))/norm_ann   
    
    print("K = {:4.4f} MeV".format(K.item()))
    print("U = {:4.4f} MeV".format(U.item()))
    print("E = {:4.4f} MeV".format((K+U).item()))

    
    # E (L=0)
    global E
    E = K+U

    return E
"""





    

