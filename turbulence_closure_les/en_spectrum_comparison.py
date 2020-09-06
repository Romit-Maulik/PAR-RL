"""
Created on Sat Sep  5 15:38:21 2020

@author: suraj
"""

import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
import os
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,ny+2] = u[:,2]
    
    u[0,:] = u[nx,:]
    u[nx+2,:] = u[2,:]
    
    return u

def laplacian(nx,ny,dx,dy,u):
    lap_bc = np.zeros((nx+3, ny+3))
    lap_bc[1:nx+2,1:ny+2] = (1.0/dx**2)*(u[2:nx+3,1:ny+2]-2.0*u[1:nx+2,1:ny+2]+u[0:nx+1,1:ny+2]) \
        + (1.0/dy**2)*(u[1:nx+2,2:ny+3]-2.0*u[1:nx+2,1:ny+2]+u[1:nx+2,0:ny+1])
       
    lap_bc = bc(nx,ny,lap_bc)
        
    return lap_bc

def energy_spectrum(nx,ny,dx,dy,w):
    epsilon = 1.0e-6

    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w[1:nx+1,1:ny+1]) 
    
    es =  np.empty((nx,ny))
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
    
    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
    
    en = np.zeros(n+1)
    
    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
#        for i in range(1,nx):
#            for j in range(1,ny):          
#                kk1 = np.sqrt(kx[i,j]**2 + ky[i,j]**2)
#                if ( kk1>(k-0.5) and kk1<(k+0.5) ):
#                    ic = ic+1
#                    en[k] = en[k] + es[i,j]
                    
        en[k] = en[k]/ic
        
    return en, n

def coarsen(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field along with the size of the data 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : solution field on coarse grid [nxc , nyc]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
    
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc [0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]     
    ufc [int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)] 
    ufc [0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):] 
    ufc [int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc = ufc*(nxc*nyc)/(nx*ny)
    
    utc = np.real(np.fft.ifft2(ufc))
    
    uc = np.zeros((nxc+1,nyc+1))
    uc[0:nxc,0:nyc] = utc
    uc[:,nyc] = uc[:,0]
    uc[nxc,:] = uc[0,:]
    uc[nxc,nyc] = uc[0,0]
        
    return uc

#%%
re = 2000.0
nx = 256
ny = 256
nxc = 32
nyc = 32
k0 = 5.0

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)
dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

folder = 'data_'+str(int(re)) + '_' + str(nx) + '_' + str(ny)
filename = './results/'+folder+'/plotting.npz'
data_fine = np.load(filename)

folder = 'data_'+str(int(re)) + '_' + str(nxc) + '_' + str(nyc)
filename = './results/'+folder+'/plotting.npz'
data_coarse = np.load(filename)

kf = data_fine['k']
en0f = data_fine['en0']
en2f = data_fine['en2']
en4f = data_fine['en4']

kc = data_coarse['k']
en0c = data_coarse['en0']
en2c = data_coarse['en2']
en4c = data_coarse['en4']

#%%
fig, axs = plt.subplots(1,1,figsize=(6,5))

c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
ese = c*(kf**4)*np.exp(-(kf/k0)**2)

line = 100*kf**(-3.0)

#axs[1,1].loglog(k,ese[:],'k', lw = 2, label='Exact')
#axs.loglog(kf,en0f[1:],'r', ls = '-', lw = 2, label='$t = 0.0$')
#axs.loglog(kf,en2f[1:], 'b', lw = 2, label = '$t = 2.0$')
axs.loglog(kf,en4f[1:], 'b', lw = 2, label = 'Fine')
axs.loglog(kf,line, 'k--', lw = 2)

#axs.loglog(kc,en0c[1:],'r', ls = '-', lw = 2, label='$t = 0.0$')
#axs.loglog(kc,en2c[1:], 'b', lw = 2, label = '$t = 2.0$')
axs.loglog(kc,en4c[1:], 'r', lw = 2, label = 'Coarse')

axs.set_xlabel('$k$')
axs.set_ylabel('$E(k)$')
axs.legend(loc=0)
axs.set_ylim(1e-8,1e-0)
axs.text(0.6, 0.8, '$k^{-3}$', transform=axs.transAxes, fontsize=16, fontweight='bold', va='top')


fig.tight_layout() 
plt.show()
fig.savefig(f'en_spectrum_{re}_{nx}_{nxc}.pdf')

#%%
folder = 'data_'+str(int(re)) + '_' + str(nx) + '_' + str(ny)
filename = './results/'+folder+'/results_s.npz'
data_s = np.load(filename)
s = data_s['sall'][-1,:,:]

w = -laplacian(nx,ny,dx,dy,s)
wc = np.zeros((nxc+3,nyc+3))
wc[1:nxc+2,1:nyc+2] = coarsen(nx,ny,nxc,nyc,w[1:nx+2,1:ny+2])
wc = bc(nxc,nyc,wc)

enf, nf = energy_spectrum(nx,ny,dx,dy,w)
enc, nc = energy_spectrum(nxc,nyc,dxc,dyc,wc)

fig, axs = plt.subplots(1,1,figsize=(6,5))

axs.loglog(kf,enf[1:], 'b', lw = 2, label = 'Fine')
axs.loglog(kf,line, 'k--', lw = 2)
axs.loglog(kc,enc[1:], 'r', lw = 2, label = 'Coarse')

axs.set_xlabel('$k$')
axs.set_ylabel('$E(k)$')
axs.legend(loc=0)
axs.set_ylim(1e-8,1e-0)
axs.text(0.6, 0.8, '$k^{-3}$', transform=axs.transAxes, fontsize=16, fontweight='bold', va='top')


fig.tight_layout() 
plt.show()


