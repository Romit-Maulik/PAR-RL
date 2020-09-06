"""
Created on Sat Sep  5 14:27:31 2020

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
from utils import *

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
# fast poisson solver using second-order central difference scheme
def fps(nx, ny, dx, dy, f):
    epsilon = 1.0e-6
    aa = -2.0/(dx*dx) - 2.0/(dy*dy)
    bb = 2.0/(dx*dx)
    cc = 2.0/(dy*dy)
    hx = 2.0*np.pi/np.float64(nx)
    hy = 2.0*np.pi/np.float64(ny)
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[:] = hx*np.float64(np.arange(0, nx))

    ky[:] = hy*np.float64(np.arange(0, ny))
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(np.cos(kx), np.cos(ky), indexing='ij')
    
    data = np.empty((nx,ny), dtype='complex128')
    data1 = np.empty((nx,ny), dtype='complex128')
    
    data[:,:] = np.vectorize(complex)(f[1:nx+1,1:ny+1],0.0)

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    e = fft_object(data)
    #e = pyfftw.interfaces.scipy_fftpack.fft2(data)
    
    e[0,0] = 0.0
    
    data1[:,:] = e[:,:]/(aa + bb*kx[:,:] + cc*ky[:,:])

    ut = np.real(fft_object_inv(data1))
    
    #periodicity
    u = np.empty((nx+3,ny+3)) 
    u[1:nx+1,1:ny+1] = ut
    u[:,ny+1] = u[:,1]
    u[nx+1,:] = u[1,:]
    u[nx+1,ny+1] = u[1,1]
    return u

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

#%%
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
def smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[0:nx+1,0:ny+1] + s[2:nx+3,2:ny+3] \
                                             -s[2:nx+3,0:ny+1] - s[0:nx+1,2:ny+3])
    
    dsdxx = (1.0/(dx*dx))*(s[2:nx+3,1:ny+2] - 2.0*s[1:nx+2,1:ny+2] \
                                         +s[0:nx+1,1:ny+2])
    
    dsdyy = (1.0/(dy*dy))*(s[1:nx+2,2:ny+3] - 2.0*s[1:nx+2,1:ny+2] \
                                         +s[1:nx+2,0:ny+1])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev
    
#%%
# compute jacobian using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def jacobian(nx,ny,dx,dy,re,w,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    jac = (j1+j2+j3)*hh
    
    return jac
    
    
#%% 
# compute rhs using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhs(nx,ny,dx,dy,re,w,s):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+3,ny+3))
    
    #Arakawa
    j1 = gg*( (w[2:nx+3,1:ny+2]-w[0:nx+1,1:ny+2])*(s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1]) \
             -(w[1:nx+2,2:ny+3]-w[1:nx+2,0:ny+1])*(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2]))

    j2 = gg*( w[2:nx+3,1:ny+2]*(s[2:nx+3,2:ny+3]-s[2:nx+3,0:ny+1]) \
            - w[0:nx+1,1:ny+2]*(s[0:nx+1,2:ny+3]-s[0:nx+1,0:ny+1]) \
            - w[1:nx+2,2:ny+3]*(s[2:nx+3,2:ny+3]-s[0:nx+1,2:ny+3]) \
            + w[1:nx+2,0:ny+1]*(s[2:nx+3,0:ny+1]-s[0:nx+1,0:ny+1]))

    j3 = gg*( w[2:nx+3,2:ny+3]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[0:nx+1,0:ny+1]*(s[0:nx+1,1:ny+2]-s[1:nx+2,0:ny+1]) \
            - w[0:nx+1,2:ny+3]*(s[1:nx+2,2:ny+3]-s[0:nx+1,1:ny+2]) \
            + w[2:nx+3,0:ny+1]*(s[2:nx+3,1:ny+2]-s[1:nx+2,0:ny+1]) )

    jac = (j1+j2+j3)*hh
    
    lap = aa*(w[2:nx+3,1:ny+2]-2.0*w[1:nx+2,1:ny+2]+w[0:nx+1,1:ny+2]) \
        + bb*(w[1:nx+2,2:ny+3]-2.0*w[1:nx+2,1:ny+2]+w[1:nx+2,0:ny+1])
    
    #call Smagorinsky model       
    #cs = 0.18
    #ev = smag(nx,ny,dx,dy,s,cs)
    
    #Central difference for Laplacian
    # f[1:nx+2,1:ny+2] = -jac + lap/re + ev*lap if using eddy viscosity model for LES
    
    f[1:nx+2,1:ny+2] = -jac + lap/re 
                        
    return f

#%%
# set initial condition for decay of turbulence problem
def decay_ic(nx,ny,dx,dy,k0):
    w = np.empty((nx+3,ny+3))
    
    epsilon = 1.0e-6
    
    kx = np.empty(nx)
    ky = np.empty(ny)
    
    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]
    
    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    ksi = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    eta = 2.0*np.pi*np.random.random_sample((int(nx/2+1), int(ny/2+1)))
    
    phase = np.zeros((nx,ny), dtype='complex128')
    wf =  np.empty((nx,ny), dtype='complex128')
    
    phase[1:int(nx/2),1:int(ny/2)] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] +
                                    eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,1:int(ny/2)] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] +
                                            eta[1:int(nx/2),1:int(ny/2)]))

    phase[1:int(nx/2),ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]), np.sin(ksi[1:int(nx/2),1:int(ny/2)] -
                                           eta[1:int(nx/2),1:int(ny/2)]))

    phase[nx-1:int(nx/2):-1,ny-1:int(ny/2):-1] = np.vectorize(complex)(np.cos(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                 eta[1:int(nx/2),1:int(ny/2)]), np.sin(-ksi[1:int(nx/2),1:int(ny/2)] -
                                                eta[1:int(nx/2),1:int(ny/2)]))

    c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
    
    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es = c*(kk**4)*np.exp(-(kk/k0)**2)
    wf[:,:] = np.sqrt((kk*es/np.pi)) * phase[:,:]*(nx*ny)
            
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')
    ut = np.real(fft_object_inv(wf)) 
    
    #w = np.zeros((nx+3,ny+3))
    
    #periodicity
    w = np.empty((nx+3,ny+3)) 
    w[1:nx+1,1:ny+1] = ut
    w[:,ny+1] = w[:,1]
    w[nx+1,:] = w[1,:]
    w[nx+1,ny+1] = w[1,1] 
    
    w = bc(nx,ny,w)    
    
    return w

#%%
# compute the energy spectrum numerically
def energy_spectrum(nx,ny,w):
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

#%% 
# read input file
l1 = []
with open('input.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ich = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])

k0 = np.float64(l1[12][0])

freq = int(nt/ns)

if (ich != 19):
    print("Check input.txt file")

#%% 
# assign parameters
nx = nd
ny = nd

nxc = ndc
nyc = ndc

#%%
pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

ifile = 0
time = 0.0

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)

x, y = np.meshgrid(x, y, indexing='ij')

#%% 
# allocate the vorticity and streamfunction arrays
w = np.empty((nx+3,ny+3)) 
s = np.empty((nx+3,ny+3))

t = np.empty((nx+3,ny+3))

r = np.empty((nx+3,ny+3))

#%%
# set the initial condition based on the problem selected
sample_ic = False

if sample_ic:
    w0 = decay_ic(nx,ny,dx,dy,k0)
else:
    nxf = 256
    nyf = 256
    dxf = lx/np.float64(nxf)
    dyf = ly/np.float64(nyf)

    folder = 'data_'+str(int(re)) + '_' + str(nxf) + '_' + str(nyf)
    filename = './results/'+folder+'/results_s.npz'
    data_s = np.load(filename)
    
    sf = data_s['sall'][0,:,:]    
    wf = -laplacian(nxf,nyf,dxf,dyf,sf)

    filename = './results/'+folder+'/results_s.npz'
    wc = np.zeros((nx+3,ny+3))
    wc[1:nx+2,1:ny+2] = coarsen(nxf,nyf,nx,ny,wf[1:nxf+2,1:nyf+2])
    wc = bc(nx,ny,wc)
    
    w0 = np.copy(wc)
    
sall = np.zeros((ns+1,nx+3,ny+3))

npx = 4
npy = 4
shift = 0.5

xp = np.arange(0,npx)*int(nx/npx) + int(shift*nx/npx)
yp = np.arange(0,npy)*int(ny/npy) + int(shift*ny/npy)
xprobe, yprobe = np.meshgrid(xp,yp)

sprobe = np.zeros((nt+1,npx,npy))
wprobe = np.zeros((nt+1,npx,npy))


k = 0    
w = np.copy(w0)
s = fps(nx, ny, dx, dy, -w)
s = bc(nx,ny,s)

sall[k,:,:] = s

wprobe[k,:,:] = w[1:nx+2,1:ny+2][xprobe, yprobe]
sprobe[k,:,:] = s[1:nx+2,1:ny+2][xprobe, yprobe]

nxf = 256
nyf = 256
folder = 'data_'+str(int(re)) + '_' + str(nxf) + '_' + str(nyf)
filename = './results/'+folder+'/probe_s_w.npz'
data = np.load(filename)

wprobe_fine = data['wprobe']
sprobe_fine = data['sprobe']

#%%
# time integration using third-order Runge Kutta method
aa = 1.0/3.0
bb = 2.0/3.0
clock_time_init = tm.time()
for k in range(1,nt+1):
    time = time + dt
    r = rhs(nx,ny,dx,dy,re,w,s)
    
    #stage-1
    t[1:nx+2,1:ny+2] = w[1:nx+2,1:ny+2] + dt*r[1:nx+2,1:ny+2]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,t,s)
    
    #stage-2
    t[1:nx+2,1:ny+2] = 0.75*w[1:nx+2,1:ny+2] + 0.25*t[1:nx+2,1:ny+2] + 0.25*dt*r[1:nx+2,1:ny+2]
    
    t = bc(nx,ny,t)
    
    s = fps(nx, ny, dx, dy, -t)
    s = bc(nx,ny,s)
    
    r = rhs(nx,ny,dx,dy,re,t,s)
    
    #stage-3
    w[1:nx+2,1:ny+2] = aa*w[1:nx+2,1:ny+2] + bb*t[1:nx+2,1:ny+2] + bb*dt*r[1:nx+2,1:ny+2]
    
    w = bc(nx,ny,w)
    
    s = fps(nx, ny, dx, dy, -w)
    s = bc(nx,ny,s)
    
    if (k%freq == 0):
        #u,v = compute_velocity(nx,ny,dx,dy,s)
        #compute_stress(nx,ny,nxc,nyc,dxc,dyc,u,v,k,freq)
        #write_data(nx,ny,dx,dy,nxc,nyc,dxc,dyc,w,s,k,freq)
        sall[int(k/freq),:,:] = s
        #export_data(nx,ny,re,int(k/freq),w,s)
        #print(k, " ", time, ' ', np.max(w))
    
    if (k%100 == 0):
        print(k, ' ', np.linalg.norm(w[1:nx+2,1:ny+2][xprobe, yprobe] - wprobe_fine[k,:,:]))
    
    wprobe[k,:,:] = w[1:nx+2,1:ny+2][xprobe, yprobe]
    sprobe[k,:,:] = s[1:nx+2,1:ny+2][xprobe, yprobe]

total_clock_time = tm.time() - clock_time_init
print('Total clock time=', total_clock_time)


#%%
# compute the exact, initial and final energy spectrum
s2 = sall[int(ns/2),:,:]
s4 = sall[-1,:,:]

w2 = -laplacian(nx,ny,dx,dy,s2)
w4 = -laplacian(nx,ny,dx,dy,s4)

#%%
fig, axs = plt.subplots(2,2,figsize=(10,9))

cs = axs[0,0].contourf(x,y,w0[1:nx+2,1:ny+2].T, 60, cmap = 'jet', interpolation='bilinear')
axs[0,0].set_xlabel('$x$')
axs[0,0].set_ylabel('$y$')
axs[0,0].set_title('$t = 0.0$')
divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(cs,cax=cax,orientation='vertical')

cs = axs[0,1].contourf(x,y,w2[1:nx+2,1:ny+2].T, 60, cmap = 'jet', interpolation='bilinear')
axs[0,1].set_xlabel('$x$')
axs[0,1].set_ylabel('$y$')
axs[0,1].set_title('$t = 2.0$')   
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(cs,cax=cax,orientation='vertical')

cs = axs[1,0].contourf(x,y,w4[1:nx+2,1:ny+2].T, 60, cmap = 'jet', interpolation='bilinear')
axs[1,0].set_xlabel('$x$')
axs[1,0].set_ylabel('$y$')
axs[1,0].set_title('$t = 4.0$')
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes('right', size='5%', pad=0.1)
fig.colorbar(cs,cax=cax,orientation='vertical')

en0, n = energy_spectrum(nx,ny,w0)
en2, n = energy_spectrum(nx,ny,w2)
en4, n = energy_spectrum(nx,ny,w4)
k = np.linspace(1,n,n)

c = 4.0/(3.0*np.sqrt(np.pi)*(k0**5))           
ese = c*(k**4)*np.exp(-(k/k0)**2)

line = 100*k**(-3.0)

#axs[1,1].loglog(k,ese[:],'k', lw = 2, label='Exact')
axs[1,1].loglog(k,en0[1:],'r', ls = '-', lw = 2, label='$t = 0.0$')
axs[1,1].loglog(k,en2[1:], 'b', lw = 2, label = '$t = 2.0$')
axs[1,1].loglog(k,en4[1:], 'y', lw = 2, label = '$t = 4.0$')
axs[1,1].loglog(k,line, 'k--', lw = 2)

axs[1,1].set_xlabel('$k$')
axs[1,1].set_ylabel('$E(k)$')
axs[1,1].legend(loc=0)
axs[1,1].set_ylim(1e-8,1e-0)
axs[1,1].text(0.8, 0.8, '$k^{-3}$', transform=axs[1,1].transAxes, fontsize=16, fontweight='bold', va='top')


fig.tight_layout() 
plt.show()
fig.savefig(f'solution_{re}_{nx}.pdf')

#%%
folder = 'data_'+str(int(re)) + '_' + str(nx) + '_' + str(ny)

if not os.path.exists('./results/'+folder):
        os.makedirs('./results/'+folder)
        
filename = './results/'+folder+'/results_s.npz'
np.savez(filename,sall = sall) 

filename = './results/'+folder+'/plotting.npz'
np.savez(filename,
         k = k, 
         line = line,
         en0 = en0,
         en2 = en2, 
         en4 = en4,
         w = w0,
         w2 = w2, 
         w4 = w4)

filename = './results/'+folder+'/probe_s_w.npz'
np.savez(filename,
         sprobe = sprobe, 
         wprobe = wprobe)