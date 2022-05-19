# -*- coding: utf-8 -*-

import torch

def gl64(w_i,function):
    """Integrates a function given via tensor by Gauss-Legendre"""
    return torch.dot(function,w_i)

def gl64_function(function,a,b,x_i,w_i):
    """Returns the quadrature a function in an interval (a,b) given via tensor 
    using the method of Gauss-Legendre with N=64"""
    
    # Parameters
    a = torch.tensor(a)
    b = torch.tensor(b)
    k1 = (b-a)/2
    k2 = (b+a)/2
    gl64 = torch.tensor(0.)
    c = 0

    for i in range(64):
        w_k = w_i[c]
        x_k = k1*x_i[c]+k2
        gl64 = gl64 + w_k*function(x_k.unsqueeze(0))
        c += 1
        
    return gl64*k1
