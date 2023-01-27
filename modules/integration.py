# -*- coding: utf-8 -*-
######################## IMPORTS ########################
import torch

######################## FUNCTIONS ########################
def gl64(w_i, function):
    """
    Performs the PyTorch dot product. It is intended to be a proxy for a 
    Gauss-Legendre integration of a function. 

    Parameters
    ----------
    w_i : tensor
        Gauss-Legendre integration weights.
    function : tensor
        Function given as a set of points.

    Returns
    -------
    tensor
        Function 'function' integrated with the weights 'w_i'.

    """
    return torch.dot(function, w_i)

def gl64_function(function, a, b, x_i, w_i):
    """
    Returns the quadrature a function in an interval (a,b) given via tensor 
    using the method of Gauss-Legendre with N=64

    Parameters
    ----------
    function : tensor
        Function to integrate.
    a : float
        lower limit.
    b : float
        upper limit.
    x_i : tensor
        Points where the function is to be evaluated.
    w_i : tensor
        Weight of the Gauss-Legendre integration method.

    Returns
    -------
    tensor
        Function 'function' integrated from 'a' to 'b'.

    """
    
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
