"""
This script computes the different possible architectures for ANNs with n hidden
layers that have the same number of parameters as a single-layer ANN.
"""

import math
from itertools import combinations_with_replacement

def equal_layers_no_bias():
    print("\nEqual layers, no bias")
    print("Legend: {n: #Hd}")
    for i in range (100):
        Hs = i+1
        dic = {}
        for n in range(100):
            if n >=2:
                p1 = (-2+2*math.sqrt(1+(n-1)*Hs))/(n-1)
                p2 = (-2-2*math.sqrt(1+(n-1)*Hs))/(n-1)
                if p1.is_integer() and p1 > 0:
                    dic[n] = int(p1)
                if p2.is_integer() and p2 > 0:
                    dic[n] = int(p2)
        print(f"#Hs = {Hs} --> {dic}")
    

def equal_layers_with_bias():
    print("\nEqual layers, with bias")
    print("Legend: {n: #Hd}")
    for i in range (100):
        Hs = i+1
        dic = {}
        for n in range(100):
            if n >=2:
                p1 = ((-n-3)+math.sqrt((n+3)**2-8*(n-1)*(1-2*Hs)))/(2*(n-1))
                p2 = ((-n-3)-math.sqrt((n+3)**2-8*(n-1)*(1-2*Hs)))/(2*(n-1))
                if p1.is_integer() and p1 > 0:
                    dic[n] = int(p1)
                if p2.is_integer() and p2 > 0:
                    dic[n] = int(p2)
        print(f"\n#Hs = {Hs} --> {dic}")


def different_layers_no_bias():
        print('\nDifferent layers, no bias')
        print('Legend: [#H1, #H2, ..., #Hn]\n')
        for i in range(15,80):
            Hs = i+1
            print(f"\n#Hs = {Hs}\n-----------------------------------")
            for k in range(9):
                n = k+2
                vectors = combinations_with_replacement([i for i in range(1,21)],n)
                good_vecs = []
                for vec in list(vectors):
                    sumatori = 0
                    for v in range(0,len(vec)-1):
                        sumatori += vec[v]*vec[v+1]
                    if 4*Hs == 2*(vec[0]+vec[n-1]) + sumatori:
                        good_vecs.append(vec)
                if len(good_vecs) > 0:
                    print(f"n = {n} --> {good_vecs[-1]}")
                    
equal_layers_no_bias()
equal_layers_with_bias()
different_layers_no_bias()
            
            
        
