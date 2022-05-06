#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:22:07 2022

@author: jozalen
"""
import torch
from torch import nn

def show_layers(model):
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:100]} \n")
        
class sc_1(nn.Module):
    def __init__(self,Nin,Nhid_prime,Nout,W1,Ws2,B,W2,Wd2):       
        super(sc_1, self).__init__()
         
        # We set the operators 
        self.lc1 = nn.Linear(Nin,Nhid_prime,bias=True) # shape = (Nhid,Nin)
        self.actfun = nn.Sigmoid() # activation function
        self.lc2 = nn.Linear(Nhid_prime,Nout,bias=False) # shape = (Nout,Nhid)
        
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
    
    
class sc_2(nn.Module):
    """The ANN takes the layer configuration as an input via a list (Layers)"""

    # Constructor
    def __init__(self,Nin,Nhid_prime,Nout,W1,Ws2,B,W2,Wd2):
        super(sc_2, self).__init__()
        self.hidden = nn.ModuleList()
        self.actfun = nn.Sigmoid()
        Layers = [Nin,Nhid_prime,Nhid_prime,Nout]
        c = 0
        for input_size, output_size in zip(Layers, Layers[1:]):
            if c == 0:
                self.hidden.append(nn.Linear(input_size, output_size, bias=True))
            else:
                self.hidden.append(nn.Linear(input_size, output_size, bias=False))
            c += 1

        # We set the parameters
        with torch.no_grad():
            self.hidden[0].bias = nn.Parameter(B)

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = self.actfun(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation
    
class sd_1(nn.Module):
    def __init__(self,Nin,Nhid_prime,Nout,W1,Ws2,B,W2,Wd2):
        super(sd_1, self).__init__()
        
        # S-state wf
        self.lcs1 = nn.Linear(Nin,Nhid_prime,bias=True) # shape = (Nhid/2,Nin)
        self.lcs2 = nn.Linear(Nhid_prime,1,bias=False) # shape = (Nout,Nhid/2)
        
        # D-state wf
        self.lcd1 = nn.Linear(Nin,Nhid_prime,bias=True) # shape = (Nhid/2,Nin)
        self.lcd2 = nn.Linear(Nhid_prime,1,bias=False) # shape = (Nout,Nhid/2)
        
        self.actfun = nn.Sigmoid() 
        
        # We set the parameters 
        with torch.no_grad():
            # S-state
            self.lcs1.bias = nn.Parameter(B)
            self.lcs1.weight = nn.Parameter(W1)
            self.lcs2.weight = nn.Parameter(Ws2)
                     
            # D-state
            self.lcd1.bias = nn.Parameter(B)
            self.lcd1.weight = nn.Parameter(W1)
            self.lcd2.weight = nn.Parameter(Wd2)
              
    # We set the architecture
    def forward(self, x): 
        o_s = self.lcs2(self.actfun(self.lcs1(x)))
        o_d = self.lcd2(self.actfun(self.lcd1(x)))
        return o_s.squeeze(),o_d.squeeze()
    
    
class sd_2(nn.Module):
    """The ANN takes the layer configuration as an input via a list (Layers)"""
    # Constructor
    def __init__(self,Nin,Nhid_prime,Nout,W1,Ws2,B,W2,Wd2):
        super(sd_2, self).__init__()
        self.hidden_s = nn.ModuleList()
        self.hidden_d = nn.ModuleList() 
        self.actfun = nn.Sigmoid()
        Layers = [Nin,Nhid_prime,Nhid_prime,Nout]
        c = 0
        for input_size, output_size in zip(Layers, Layers[1:]):
            if c == 0:
                self.hidden_s.append(nn.Linear(input_size, output_size, bias=True))
                self.hidden_d.append(nn.Linear(input_size, output_size, bias=True))
            else:
                self.hidden_s.append(nn.Linear(input_size, output_size, bias=False))
                self.hidden_d.append(nn.Linear(input_size, output_size, bias=False))
            c += 1

        # We configure the parameters
        with torch.no_grad():
            self.hidden_s[0].bias = nn.Parameter(B)
            self.hidden_d[0].bias = nn.Parameter(B)

    # Prediction
    def forward(self, activation):
        L = len(self.hidden_s)
        activation_s = activation
        activation_d = activation
        
        # S-state VANN
        for (l, linear_transform) in zip(range(L), self.hidden_s):
            if l < L - 1:
                activation_s = self.actfun(linear_transform(activation_s))
            else:
                activation_s = linear_transform(activation_s)
        
        # D-state VANN
        for (l, linear_transform) in zip(range(L), self.hidden_d):
            if l < L - 1:
                activation_d = self.actfun(linear_transform(activation_d))
            else:
                activation_d = linear_transform(activation_d)
        
        return activation_s.squeeze(),activation_d.squeeze()
    