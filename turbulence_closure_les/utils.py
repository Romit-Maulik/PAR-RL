#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:45:02 2020

@author: suraj
"""
import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os
from numba import jit

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
@jit
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,1] = u[:,ny+1]
    u[:,ny+3] = u[:,3]
    u[:,ny+4] = u[:,4]
    
    u[0,:] = u[nx,:]
    u[1,:] = u[nx+1,:]
    u[nx+3,:] = u[3,:]
    u[nx+4,:] = u[4,:]
    
    return u

#%%
# fast poisson solver using second-order central difference scheme 
def fpsd(nx, ny, dx, dy, f):
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
    
    data[:,:] = np.vectorize(complex)(f[0:nx,0:ny],0.0)

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
    u = np.empty((nx+1,ny+1)) 
    u[0:nx,0:ny] = ut
    u[:,ny] = u[:,0]
    u[nx,:] = u[0,:]
    u[nx,ny] = u[0,0]
    
    return u

#%%
@jit
def les_filter(nx,ny,nxc,nyc,u):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx+1, ny+1]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0 
    utc = np.real(np.fft.ifft2(uf))
    
    uc = np.zeros((nx+1,ny+1))
    uc[0:nx,0:ny] = utc
    
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    
    return uc

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx (size = [nx+1,ny+1])
    uy : du/dy (size = [nx+1,ny+1])
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy

#%%
def dyn_smag(nx,ny,kappa,sc,wc):
    '''
    compute the eddy viscosity using Germanos dynamics procedure with Lilys 
    least square approximation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    kapppa : sub-filter grid filter ratio
    wc : vorticity on LES grid
    sc : streamfunction on LES grid
    
    Output
    ------
    ev : (cs*delta)**2*|S| (size = [nx+1,ny+1])
    '''
    
    nxc = int(nx/kappa) 
    nyc = int(ny/kappa)
    
    scc = les_filter(nx,ny,nxc,nyc,sc[2:nx+3,2:ny+3])
    wcc = les_filter(nx,ny,nxc,nyc,wc[2:nx+3,2:ny+3])
    
    scx,scy = grad_spectral(nx,ny,sc[2:nx+3,2:ny+3])
    wcx,wcy = grad_spectral(nx,ny,wc[2:nx+3,2:ny+3])
    
    wcxx,wcxy = grad_spectral(nx,ny,wcx)
    wcyx,wcyy = grad_spectral(nx,ny,wcy)
    
    scxx,scxy = grad_spectral(nx,ny,scx)
    scyx,scyy = grad_spectral(nx,ny,scy)
    
    dac = np.sqrt(4.0*scxy**2 + (scxx - scyy)**2) # |\bar(s)|
    dacc = les_filter(nx,ny,nxc,nyc,dac)        # |\tilde{\bar{s}}| = \tilde{|\bar(s)|}
    
    sccx,sccy = grad_spectral(nx,ny,scc)
    wccx,wccy = grad_spectral(nx,ny,wcc)
    
    wccxx,wccxy = grad_spectral(nx,ny,wccx)
    wccyx,wccyy = grad_spectral(nx,ny,wccy)
    
    scy_wcx = scy*wcx
    scx_wcy = scx*wcy
    
    scy_wcx_c = les_filter(nx,ny,nxc,nyc,scy_wcx)
    scx_wcy_c = les_filter(nx,ny,nxc,nyc,scx_wcy)
    
    h = (sccy*wccx - sccx*wccy) - (scy_wcx_c - scx_wcy_c)
    
    t = dac*(wcxx + wcyy)
    tc = les_filter(nx,ny,nxc,nyc,t)
    
    m = kappa**2*dacc*(wccxx + wccyy) - tc
    
    hm = h*m
    mm = m*m
    
    CS2 = (np.sum(0.5*(hm + abs(hm)))/np.sum(mm))
    
    ev = CS2*dac
    
    return ev

#%%
def stat_smag(nx,ny,dx,dy,s,cs):
        
        
    dsdxy = (1.0/(4.0*dx*dy))*(s[1:nx+2,1:ny+2] + s[3:nx+4,3:ny+4] \
                                             -s[3:nx+4,1:ny+2] - s[1:nx+2,3:ny+4])
    
    dsdxx = (1.0/(dx*dx))*(s[3:nx+4,2:ny+3] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[1:nx+2,2:ny+3])
    
    dsdyy = (1.0/(dy*dy))*(s[2:nx+3,3:ny+4] - 2.0*s[2:nx+3,2:ny+3] \
                                         +s[2:nx+3,1:ny+2])
    
    ev = cs*cs*dx*dy*np.sqrt(4.0*dsdxy*dsdxy + (dsdxx-dsdyy)*(dsdxx-dsdyy))
    
    return ev

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
# compute jacobian using arakawa scheme
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def jacobian(nx,ny,dx,dy,re,w,s):
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    # Arakawa
    j1 = gg*( (w[3:nx+4,2:ny+3]-w[1:nx+2,2:ny+3])*(s[2:nx+3,3:ny+4]-s[2:nx+3,1:ny+2]) \
             -(w[2:nx+3,3:ny+4]-w[2:nx+3,1:ny+2])*(s[3:nx+4,2:ny+3]-s[1:nx+2,2:ny+3]))

    j2 = gg*( w[3:nx+4,2:ny+3]*(s[3:nx+4,3:ny+4]-s[3:nx+4,1:ny+2]) \
            - w[1:nx+2,2:ny+3]*(s[1:nx+2,3:ny+4]-s[1:nx+2,1:ny+2]) \
            - w[2:nx+3,3:ny+4]*(s[3:nx+4,3:ny+4]-s[1:nx+2,3:ny+4]) \
            + w[2:nx+3,1:ny+2]*(s[3:nx+4,1:ny+2]-s[1:nx+2,1:ny+2]))

    j3 = gg*( w[3:nx+4,3:ny+4]*(s[2:nx+3,3:ny+4]-s[3:nx+4,2:ny+3]) \
            - w[1:nx+2,1:ny+2]*(s[1:nx+2,2:ny+3]-s[2:nx+3,1:ny+2]) \
            - w[1:nx+2,3:ny+4]*(s[2:nx+3,3:ny+4]-s[1:nx+2,2:ny+3]) \
            + w[3:nx+4,1:ny+2]*(s[3:nx+4,2:ny+3]-s[2:nx+3,1:ny+2]) )

    jac = (j1+j2+j3)*hh
    
    return jac

#%%
# compute the energy spectrum numerically
def energy_spectrumd(nx,ny,dx,dy,w):
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
    wf = fft_object(w[0:nx,0:ny]) 
    
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
# compute rhs using arakawa scheme (formulas are based on one ghost points, 
# borrowed from fdm solver)
# computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
# no ghost points
def rhsa(nx,ny,dx,dy,re,we,se):
    aa = 1.0/(dx*dx)
    bb = 1.0/(dy*dy)
    gg = 1.0/(4.0*dx*dy)
    hh = 1.0/3.0
    
    f = np.zeros((nx+5,ny+5))
    
    w = we[1:nx+4,1:ny+4]
    s = se[1:nx+4,1:ny+4]
    
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
    
    f[2:nx+3,2:ny+3] = -jac + lap/re 
                        
    return f
    
#%% serial compact schemes
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using Thomas algorithm
#-----------------------------------------------------------------------------#
def tdms(a,b,c,r,s,e):
    gam = np.zeros((e+1))
    u = np.zeros((e+1))
    
    bet = b[s]
    u[s] = r[s]/bet
    
    for i in range(s+1,e+1):
        gam[i] = c[i-1]/bet
        bet = b[i] - a[i]*gam[i]
        u[i] = (r[i] - a[i]*u[i-1])/bet
    
    for i in range(e-1,s-1,-1):
        u[i] = u[i] - gam[i+1]*u[i+1]
    
    return u
        
#-----------------------------------------------------------------------------#
# Solution to tridigonal system using cyclic Thomas algorithm
#-----------------------------------------------------------------------------#
def ctdms(a,b,c,alpha,beta,r,s,e):
    bb = np.zeros((e+1))
    u = np.zeros((e+1))
    
    gamma = -b[s]
    bb[s] = b[s] - gamma
    bb[e] = b[e] - alpha*beta/gamma
    
#    for i in range(s+1,e):
#        bb[i] = b[i]
    
    bb[s+1:e] = b[s+1:e]
    
    x = tdms(a,bb,c,r,s,e)
    
    u[s] = gamma
    u[e] = alpha
    
    z = tdms(a,bb,c,u,s,e)
    
    fact = (x[s] + beta*x[e]/gamma)/(1.0 + z[s] + beta*z[e]/gamma)
    
#    for i in range(s,e+1):
#        x[i] = x[i] - fact*z[i]
    
    x[s:e+1] = x[s:e+1] - fact*z[s:e+1]
        
    return x

#-----------------------------------------------------------------------------#
#cu3dp: 3rd-order compact upwind scheme for the first derivative(up)
#       periodic boundary conditions (0=n), h=grid spacing
#       p: free upwind paramater suggested (p>0 for upwind)
#                                           p=0.25 in Zhong (JCP 1998)
#		
#-----------------------------------------------------------------------------#
def cu3dp(u,p,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    up = np.zeros((n+1))
    
    a[:] = 1.0 + p
    b[:] = 4.0
    c[:] = 1.0 - p

#    for i in range(1,n):
#        r[i] = ((-3.0-2.0*p)*u[i-1] + 4.0*p*u[i] + (3.0-2.0*p)*u[i+1])/h
    r[1:n] = ((-3.0-2.0*p)*u[0:n-1] + 4.0*p*u[1:n] + (3.0-2.0*p)*u[2:n+1])/h
    r[0] = ((-3.0-2.0*p)*u[n-1] + 4.0*p*u[0] + (3.0-2.0*p)*u[1])/h  
    
    alpha = 1.0 - p
    beta = 1.0 + p
    
    x = ctdms(a,b,c,alpha,beta,r,0,n-1)
    
    up[0:n] = x[0:n]
    
    up[n] = up[0]
    
    return up

#-----------------------------------------------------------------------------#
# c4dp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4dp(u,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    up = np.zeros((n+1))
    
    a[:] = 1.0/4.0
    b[:] = 1.0
    c[:] = 1.0/4.0

#    for i in range(1,n):
#        r[i] = (3.0/2.0)*(u[i+1] - u[i-1])/(2.0*h)
    r[1:n] = (3.0/2.0)*(u[2:n+1] - u[0:n-1])/(2.0*h)
    r[0] = (3.0/2.0)*(u[1] - u[n-1])/(2.0*h)
    
    alpha = 1.0/4.0
    beta = 1.0/4.0
    
    x = ctdms(a,b,c,alpha,beta,r,0,n-1)
    
    up[0:n] = x[0:n]
    
    up[n] = up[0]
    
    return up

#-----------------------------------------------------------------------------#
# c4ddp:  4th-order compact scheme for first-degree derivative(up)
#        periodic boundary conditions (0=n), h=grid spacing
#        tested
#		
#-----------------------------------------------------------------------------#
def c4ddp(u,h,n):
    a = np.zeros((n))
    b = np.zeros((n))        
    c = np.zeros((n))    
    x = np.zeros((n))
    r = np.zeros((n))  
    upp = np.zeros((n+1))
    
    a[:] = 1.0/10.0
    b[:] = 1.0
    c[:] = 1.0/10.0

#    for i in range(1,n):
#        r[i] = (6.0/5.0)*(u[i-1] - 2.0*u[i] + u[i+1])/(h*h)
    
    r[1:n] = (6.0/5.0)*(u[0:n-1] - 2.0*u[1:n] + u[2:n+1])/(h*h)
    r[0] = (6.0/5.0)*(u[n-1] - 2.0*u[0] + u[1])/(h*h)
    
    alp = 1.0/10.0
    beta = 1.0/10.0
    
    x = ctdms(a,b,c,alp,beta,r,0,n-1)
    
    upp[0:n] = x[0:n]
    
    upp[n] = upp[0]
    
    return upp      

#%% rhs
def rhs_cu3(nx,ny,dx,dy,re,pCU3,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))
    
    # compute wxx
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wxx = c4ddp(a,dx,nx)
        
        lap[2:nx+3,j] = wxx[:]
    
    # compute wyy
    for i in range(2,nx+3):
        a = w[i,2:ny+3]        
        wyy = c4ddp(a,dx,nx)
        
        lap[i,2:ny+3] = lap[i,2:ny+3] + wyy[:]
    
    # Jacobian (convective term): upwind
    
    # sy: u
    sy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = s[i,2:ny+3]        
        sy[i-2,:] = c4dp(a,dx,nx)
    
    # computation of wx
    wxp = np.zeros((nx+1,ny+1))
    wxn = np.zeros((nx+1,ny+1))
    
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wxp[:,j-2] = cu3dp(a, pCU3, dx, nx) # upwind for wx   
        wxn[:,j-2] = cu3dp(a, -pCU3, dx, nx) # downwind for wx     
    
    # upwinding
    syp = np.where(sy>0,sy,0) # max(sy[i,j],0)
    syn = np.where(sy<0,sy,0) # min(sy[i,j],0)

    # sx: -v
    sx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = s[2:nx+3,j]
        sx[:,j-2] = -c4dp(a, dx, nx)
    
    # computation of wy
    wyp = np.zeros((nx+1,ny+1))
    wyn = np.zeros((nx+1,ny+1))
    
    for i in range(2,nx+3):
        a = w[i,2:ny+3]
        wyp[i-2,:] = cu3dp(a, pCU3, dy, ny) # upwind for wy
        wyn[i-2,:] = cu3dp(a, -pCU3, dy, ny) # downwind for wy 
    
    # upwinding
    sxp = np.where(sx>0,sx,0) # max(sx[i,j],0)
    sxn = np.where(sx<0,sx,0) # min(sx[i,j],0)
    
    jac[2:nx+3,2:ny+3] = (syp*wxp + syn*wxn) + (sxp*wyp + sxn*wyn)
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re 
    
    del sy, sx, syp, syn, sxp, sxn, wxp, wxn, wyp, wyn
    
    return f

#%%
def rhs_compact(nx,ny,dx,dy,re,w,s):
    lap = np.zeros((nx+5,ny+5))
    jac = np.zeros((nx+5,ny+5))
    f = np.zeros((nx+5,ny+5))
    
    # compute wxx
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wxx = c4ddp(a,dx,nx)
        
        lap[2:nx+3,j] = wxx[:]
    
    # compute wyy
    for i in range(2,nx+3):
        a = w[i,2:ny+3]        
        wyy = c4ddp(a,dx,nx)
        
        lap[i,2:ny+3] = lap[i,2:ny+3] + wyy[:]
    
    # Jacobian (convective term): upwind
    
    # sy
    sy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = s[i,2:ny+3]        
        sy[i-2,:] = c4dp(a,dx,nx)
    
    # computation of wx
    wx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = w[2:nx+3,j]
        wx[:,j-2] = c4dp(a,dx,nx)
        
    
    # sx
    sx = np.zeros((nx+1,ny+1))
    for j in range(2,ny+3):
        a = s[2:nx+3,j]
        sx[:,j-2] = c4dp(a, dx, nx)
    
    # computation of wy
    wy = np.zeros((nx+1,ny+1))
    for i in range(2,nx+3):
        a = w[i,2:ny+3]
        wy[i-2,:] = c4dp(a, dx, nx)
    
    jac[2:nx+3,2:ny+3] = (sy*wx - sx*wy)
    
    f[2:nx+3,2:ny+3] = -jac[2:nx+3,2:ny+3] + lap[2:nx+3,2:ny+3]/re
    
    del sy, wx, sx, wy
    
    return f