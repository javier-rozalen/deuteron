import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from torch.autograd import Variable

class N3LO:

	def __init__(self, filename, filename2):
		self.hbar = 197.32968
		N = 1000

		self.k = np.zeros((N,N))
		self.kp = np.zeros((N,N))

		self.vNN_S = np.zeros((N,N))
		self.vNN_D = np.zeros((N,N))
		self.vNN_SD = np.zeros((N,N))
		self.vNN_DS = np.zeros((N,N))

		f = np.genfromtxt(filename)
		self.k_tmp = f[:,0]
		self.kp_tmp = f[:,1]
		
		for ik in range(N):
			for jk in range(N):
				self.k[ik,jk], self.kp[ik,jk], _, _, self.vNN_S[ik,jk], self.vNN_D[ik,jk], self.vNN_SD[ik,jk], self.vNN_DS[ik,jk] = f[1000*ik+jk,:]

		file2 = np.genfromtxt(filename2)
		self.ksd = file2[:,0]
		self.wfs = file2[:,1]
		self.wfd = file2[:,2]

	def getPotential(self):
		
		N=1000
		V = np.zeros(shape=(2*N,2*N))
	
		for ik in range(N):
			for jk in range(N):
				V[ik,jk] = self.vNN_S[ik,jk]*(self.hbar**3)
				V[ik+N,jk] = self.vNN_DS[ik,jk]*(self.hbar**3)
				V[ik,jk+N] = self.vNN_SD[ik,jk]*(self.hbar**3)
				V[ik+N,jk+N] = self.vNN_D[ik,jk]*(self.hbar**3)

		return torch.Tensor(V)#, torch.Tensor(self.vNN_S), torch.Tensor(self.vNN_SD), torch.Tensor(self.vNN_DS), torch.Tensor(self.vNN_D)

	def getOrbitalPotentials(self):
		return torch.Tensor(self.vNN_S)*(self.hbar**3), torch.Tensor(self.vNN_SD)*(self.hbar**3), torch.Tensor(self.vNN_DS)*(self.hbar**3), torch.Tensor(self.vNN_D)*(self.hbar**3)

	def getWavefunction(self):
		return torch.Tensor(self.ksd).unsqueeze(1), torch.Tensor(self.wfs).unsqueeze(1), torch.Tensor(self.wfd).unsqueeze(1)


