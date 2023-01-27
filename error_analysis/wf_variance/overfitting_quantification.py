# -*- coding: utf-8 -*-
############# CONTENTS OF THE FILE #############
"""
This file can be used to compute all the values in TABLE I in the article. It
does not store any data, it just prints it out to console. 

The aim of the computations performed here is to have a better estimate of the
fidelities of the models whose wave functions present fluctuations. To this end
we compute the fidelities using a new, much denser mesh so as to capture all
fluctuations that occured in-between the old mesh points.

· In Section 'DATA PREPARATION' we first fetch a uniform, dense mesh and the 
ANN wave functions with Nhid=100 hidden neurons previously computed in that 
same mesh. Then we also fetch the exact wave functions previously computed in 
the vrey same mesh. 

· In Section 'OVERLAP COMPUTATION' we compute the overlap between the ANN wave
functions fetch in the previous section and the exact wave functions. We do 
this only for the '1sd' and '2sd' architectures, and for those models with 
Nhid=100 hidden neurons. 
"""

############# IMPORTS #############
import torch, statistics, N3LO, os, sys, pathlib
from scipy import integrate
import numpy as np
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('../..')

# My modules
from modules.physical_constants import mu, hbar

############# ADJUSTABLE PARAMETERS #############
net_archs = ['1sc', '2sc', '1sd', '2sd']

############# DATA PREPARATION #############
dict_psi_s = dict()
dict_psi_s2 = dict()
dict_psi_d = dict()
dict_psi_d2 = dict() 
q_test = []
with open('../../deuteron_data/q_uniform_1000_10000/mesh.dat', 'r') as f:
    for line in f.readlines():
        try:
            q = float(line.split(' ')[2])
            q_test.append(q)
        except:
            try:
                q = float(line.split(' ')[3])
                q_test.append(q)
            except:
                pass
    f.close()
n_test = len(q_test)
q_test2 = [q**2 for q in q_test]
# print(f'q_test: {q_test}')

# ANN
for arch in net_archs:
    with open(f'plot_data/wf_{arch}.txt', 'r') as f:
        c = 0
        psi_s, psi_d, psi_s2, psi_d2 = [], [], [], []
        for line in f.readlines():
            if len(line) > 1:
                s, d = float(line.split(' ')[0]), float(line.split(' ')[1])
                psi_s.append(s)
                psi_s2.append(s**2)
                psi_d.append(d)
                psi_d2.append(d**2)
            else:
                dict_psi_s[f"{arch}_{c}"] = psi_s
                dict_psi_d[f"{arch}_{c}"] = psi_d
                dict_psi_s2[f"{arch}_{c}"] = psi_s2
                dict_psi_d2[f"{arch}_{c}"] = psi_d2
                psi_s, psi_d, psi_s2, psi_d2 = [], [], [], []
                c += 1
        f.close()

# EXACT
print('Computing the exact wave functions...\n')
n3lo = N3LO.N3LO(f"../../deuteron_data/q_uniform_{n_test}_10000/2d_vNN_J1.dat",
                 f"../../deuteron_data/q_uniform_{n_test}_10000/wfk.dat")

psi_target_s = n3lo.getWavefunction()[1].squeeze(1)
psi_target_d = n3lo.getWavefunction()[2].squeeze(1)   
psi_target_s2 = [psi**2 for psi in psi_target_s]
psi_target_d2 = [psi**2 for psi in psi_target_d]
psi_target_s_norm = integrate.simpson([i*j for i, j in zip(q_test2,
                                                          psi_target_s2)],
                                      q_test)
psi_target_d_norm = integrate.simpson([i*j for i, j in zip(q_test2,
                                                          psi_target_d2)],
                                      q_test)
psi_target_norm = psi_target_s_norm + psi_target_d_norm
psi_target_s_normalized = psi_target_s/np.sqrt(psi_target_norm)
psi_target_d_normalized = psi_target_d/np.sqrt(psi_target_norm)
targ_s = psi_target_s_normalized
targ_d = psi_target_d_normalized   
targ_s2 = [psi**2 for psi in targ_s] 
targ_d2 = [psi**2 for psi in targ_d] 

############# OVERLAP COMPUTATION #############
# COMPUTING THE OVERLAP OF THE OVERFITTED AND NON-OVERFITTED NETS WITH THE 
# EXACT FUNCTIONS
"""
Note: we use the functions with Nhid=100 1sd and 2sd
"""

sd1_fidelities_s, sd1_fidelities_d = list(), list()
sd2_fidelities_s, sd2_fidelities_d = list(), list()

c = 0
for key in list(dict_psi_s.keys()):   
    if key.split('_')[0][2] == 'd':              
        wf_s = np.array(dict_psi_s[key])
        wf_d = np.array(dict_psi_d[key])
        wf_s_norm = integrate.simpson([q*psi**2 for q, psi in zip(q_test2,
                                                                 wf_s)],q_test)
        wf_d_norm = integrate.simpson([q*psi**2 for q, psi in zip(q_test2,
                                                                 wf_d)],q_test)       
        wf_s_plot = wf_s*np.sqrt(psi_target_s_norm/(psi_target_norm*wf_s_norm))
        wf_d_plot = wf_d*np.sqrt(psi_target_d_norm/(psi_target_norm*wf_d_norm))
        
        integrable_s, integrable_d = [], []
        for i in range(0, n_test):
            integrable_s.append(wf_s[i]*psi_target_s[i]*q_test2[i])
            integrable_d.append(wf_d[i]*psi_target_d[i]*q_test2[i])
        fidelity_s =  (integrate.simpson(integrable_s, q_test))**2/ \
            (psi_target_s_norm*wf_s_norm)
        fidelity_d =  (integrate.simpson(integrable_d, q_test))**2/ \
            (psi_target_d_norm*wf_d_norm)
        
        if key.split('_')[0][0] == '1':
            sd1_fidelities_s.append(fidelity_s)
            sd1_fidelities_d.append(fidelity_d)
            c += 1
        elif key.split('_')[0][0] == '2':
            sd2_fidelities_s.append(fidelity_s)
            sd2_fidelities_d.append(fidelity_d)
            c += 1  

print('\nFIDELITIES')
print('----------------------------------------------------------------------')
print('FIDELITY VALUES OF MODEL 1 SD')
print('--------------------------------')
print('Mean fidelities_s of 1sd: {:.5f}'.format(statistics.mean(
    sd1_fidelities_s)))
print('Sigma(fidelities_s) of 1sd: {:.5f}'.format(statistics.stdev(
    sd1_fidelities_s)))
print('Mean fidelities_d of 1sd: {:.5f}'.format(statistics.mean(
    sd1_fidelities_d)))
print('Sigma(fidelities_d) of 1sd: {:.5f}'.format(statistics.stdev(
    sd1_fidelities_d)))
print('\nFIDELITY VALUES OF MODEL 2 SD')
print('--------------------------------')
print('Mean fidelities_s of 2sd: {:.5f}'.format(statistics.mean(
    sd2_fidelities_s)))
print('Sigma(fidelities_s) of 2sd: {:.5f}'.format(statistics.stdev(
    sd2_fidelities_s)))
print('Mean fidelities_d of 2sd: {:.5f}'.format(statistics.mean(
    sd2_fidelities_d)))
print('Sigma(fidelities_d) of 2sd: {:.5f}'.format(statistics.stdev(
    sd2_fidelities_d)))

############# ENERGY COMPUTATION #############
# COMPUTING E OF THE OVERFITTED AND NON-OVERFITTED NETS
"""
Note: we use the functions with Nhid=100
"""

sd1_K, sd1_K_s, sd1_K_d, sd1_V = list(), list(), list(), list()
sd2_K, sd2_K_s, sd2_K_d, sd2_V = list(), list(), list(), list()
sd1_E, sd2_E = list(), list()

q_test4 = [q**2 for q in q_test2]
V_ij = n3lo.getPotential()

def trapezoid_weights(x):
    N = len(x)
    w_k = []
    for k in range(N):
        if k == 0:
            w_k.append(0.5*(x[1]-x[0]))
        elif k == N-1:
            w_k.append(0.5*(x[-1]-x[-2]))
        else:
            w_k.append(0.5*(x[k+1]-x[k-1])) 
            
    return w_k

q_test, q_test2 = np.array(q_test), np.array(q_test2) 

q_test2_2N = torch.cat((torch.from_numpy(q_test2), torch.from_numpy(q_test2)))
q_test2_2N = [e.item() for e in q_test2_2N]

w_N = trapezoid_weights(q_test)
w_2N = torch.cat((torch.tensor(w_N), torch.tensor(w_N)))
w_2N = [e.item() for e in w_2N]

c = 0
for key in list(dict_psi_s.keys()): 
    if key.split('_')[0][2] == 'd':
        wf_s = np.array(dict_psi_s[key])
        wf_d = np.array(dict_psi_d[key])
        wf_s_norm = integrate.simpson([q*psi**2 for q, psi in zip(q_test2,
                                                                 wf_s)],q_test)
        wf_d_norm = integrate.simpson([q*psi**2 for q, psi in zip(q_test2,
                                                                 wf_d)],q_test)
        wf_s_plot = wf_s*np.sqrt(psi_target_s_norm/(psi_target_norm*wf_s_norm))
        wf_d_plot = wf_d*np.sqrt(psi_target_d_norm/(psi_target_norm*wf_d_norm))
        wf_norm = np.sqrt(wf_s_norm+wf_d_norm)

        
        # K
        wf_s2 = [psi**2 for psi in wf_s_plot]
        wf_d2 = [psi**2 for psi in wf_d_plot]
        integrable_s = [q_test4[i]*wf_s2[i] for i in range(0, n_test)]
        integrable_d = [q_test4[i]*wf_d2[i] for i in range(0, n_test)]
        K_s = (hbar**2/mu)*(integrate.simpson(integrable_s, q_test))
        K_d = (hbar**2/mu)*(integrate.simpson(integrable_d, q_test))
        K = K_s + K_d
        # K exact
        integrable_s_exact = [q_test4[i]*targ_s2[i] for i in range(0, n_test)]
        integrable_d_exact = [q_test4[i]*targ_d2[i] for i in range(0, n_test)]
        K_s_exact = (hbar**2/mu)*(integrate.simpson(integrable_s_exact,
                                                    q_test))
        K_d_exact = (hbar**2/mu)*(integrate.simpson(integrable_d_exact,
                                                    q_test))
        K_exact = K_s_exact + K_d_exact

        # V
        psi_2N = list(wf_s_plot)
        for x in wf_d_plot:
            psi_2N.append(x)
        y = [i*j*k for i, j, k in zip(w_2N, q_test2_2N, psi_2N)]
        V = np.matmul(y, np.matmul(V_ij, y)).item()
        # V exact
        psi_2N_exact = list(targ_s)
        for x in targ_d:
            psi_2N_exact.append(x) 
        y_exact = [i*j*k for i, j, k in zip(w_2N, q_test2_2N, psi_2N_exact)]
        V_exact = np.matmul(y_exact, np.matmul(V_ij, y_exact)).item()

        if key.split('_')[0][0] == '1':
            sd1_K.append(K)
            sd1_K_s.append(K_s)
            sd1_K_d.append(K_d)
            sd1_V.append(V)
            sd1_E.append(K + V)
            c += 1
        elif key.split('_')[0][0] == '2':
            sd2_K.append(K)
            sd2_K_s.append(K_s)
            sd2_K_d.append(K_d)
            sd2_V.append(V)
            sd2_E.append(K + V)
            c += 1
     
print('\n\nENERGIES')
print('----------------------------------------------------------------------')
print('EXACT VALUES DEUTERON PROGRAM')
print('--------------------------------')
print(f'K = {14.629060} MeV, V = {-16.853635} MeV, E = {-2.2245750} MeV\n')
print('EXACT VALUES COMPUTED HERE')
print('--------------------------------')
print(f'K = {K_exact} MeV, V = {V_exact} MeV, E = {K_exact+V_exact} MeV\n')
print('VALUES OF MODEL 1 SD')
print('--------------------------------')
print('Mean K of 1sd: {:.5f} MeV'.format(statistics.mean(sd1_K)))
print('Sigma(K) of 1sd: {:.5f} MeV'.format(statistics.stdev(sd1_K)))
print('Mean V of 1sd: {:.5f} MeV'.format(statistics.mean(sd1_V)))
print('Sigma(V) of 1sd: {:.5f} MeV'.format(statistics.stdev(sd1_V)))
print('Mean E of 1sd: {:.5f} MeV'.format(statistics.mean(sd1_E)))
print('Sigma(E) of 1sd: {:.5f} MeV'.format(statistics.stdev(sd1_E)))
print('\nVALUES OF MODEL 2 SD')
print('--------------------------------')
print('Mean K of 2sd: {:.5f} MeV'.format(statistics.mean(sd2_K)))
print('Sigma(K) of 2sd: {:.5f} MeV'.format(statistics.stdev(sd2_K)))
print('Mean V of 2sd: {:.5f} MeV'.format(statistics.mean(sd2_V)))
print('Sigma(V) of 2sd: {:.5f} MeV'.format(statistics.stdev(sd2_V)))
print('Mean E of 2sd: {:.5f} MeV'.format(statistics.mean(sd2_E)))
print('Sigma(E) of 2sd: {:.5f} MeV'.format(statistics.stdev(sd2_E)))
