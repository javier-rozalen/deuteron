# -*- coding: utf-8 -*-

"""This program performs the quadrature of a function using the method of
Gauss-Legendre with N=64 points (this can be easily changed). The Legendre
Polynomials over the interval (-1,1) are used."""

import torch, numpy as np

x, w = np.polynomial.legendre.leggauss(64)
x_i = [torch.tensor(float(e)) for e in x]
w_i = [torch.tensor(float(e)) for e in w]

def gl64(function,a,b):
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
