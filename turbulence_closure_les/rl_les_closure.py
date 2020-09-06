"""
Created on Tue Jun  2 13:36:55 2020

@author: suraj

Taken from supplementary material of 
"Restoring chaos using deep reinforcement learning" Chaos 30, 031102 (2020)

"""
import gym
import os
import csv
import pyfftw
import time
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

class les_closure(gym.Env):
    
    metadata = {'render.modes': ['human'],
                'video.frames_per_second' : 30}
    
    # Initialize the parameters for the Lorenz system and the RL
    def __init__(self, env_config):
        self.n = env_config['n']
        self.path = os.getcwd()+'/les_closure.csv'
        
        l1 = []
        with open('input.txt') as f:
            for l in f:
                l1.append((l.strip()).split("\t"))
        
        self.nd = np.int64(l1[0][0])
        self.nt = np.int64(l1[1][0])
        self.re = np.float64(l1[2][0])
        self.dt = np.float64(l1[3][0])
        self.ns = np.int64(l1[4][0])
        self.isolver = np.int64(l1[5][0])
        self.isc = np.int64(l1[6][0])
        self.ich = np.int64(l1[7][0])
        self.ipr = np.int64(l1[8][0])
        self.ndc = np.int64(l1[9][0])
        self.k0 = np.float64(l1[12][0])
        
        self.nx = self.nd
        self.ny = self.nd
        
        self.time = 0.0
        self.freq = int(self.nt/self.ns)
        
        self.cs_update_freq = 10
        self.n_start = 0 
        self.n_end = 0 
        
        pi = np.pi
        self.lx = 2.0*pi
        self.ly = 2.0*pi
        self.dx = self.lx/np.float64(self.nx)
        self.dy = self.ly/np.float64(self.ny)
        
        self.nxf = 256
        self.nyf = 256
        self.dxf = self.lx/np.float64(self.nxf)
        self.dyf = self.ly/np.float64(self.nyf)
                
        self.x = np.linspace(0.0,2.0*np.pi,self.nx+1)
        self.y = np.linspace(0.0,2.0*np.pi,self.ny+1)
        self.x, self.y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.w = np.empty((self.nx+3,self.ny+3)) 
        self.s = np.empty((self.nx+3,self.ny+3))
        self.t = np.empty((self.nx+3,self.ny+3))        
        self.r = np.empty((self.nx+3,self.ny+3))
        
        npx = 4
        npy = 4
        shift = 0.5
        
        xp = np.arange(0,npx)*int(self.nx/npx) + int(shift*self.nx/npx)
        yp = np.arange(0,npy)*int(self.ny/npy) + int(shift*self.ny/npy)
        self.xprobe, self.yprobe = np.meshgrid(xp,yp)
        
        nxf = 256
        nyf = 256
        folder = 'data_'+str(int(self.re)) + '_' + str(nxf) + '_' + str(nyf)
        filename = './results/'+folder+'/probe_s_w.npz'
        data = np.load(filename)
        
        self.wprobe_fine = data['wprobe']
        self.sprobe_fine = data['sprobe']
        
        # Upper bound for control perturbation values
        low_action = np.array([0.05])
        high_action = np.array([0.5])

        # Define the unbounded state-space
        high_obs = np.finfo(np.float32).max*np.ones(16)
        
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)
        
        #Define the bounded action space
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state  = None
        self.reward = None
        
        # Stop episode after 4000 steps
        self.n_terminal = 4000
        
        # Stepwise rewards
        self.negative_reward = -10.0
        self.positive_reward = 10.0
        
        #Terminal reward
        self.term_punish =  self.negative_reward*10.
        self.term_mean = self.negative_reward/5.    # -2.0
        
        # To compute mean reward over the final 2000 steps
        self.n_max = self.n_terminal 
        self.rewards = []
    
    # Seed for random number generator
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # fast poisson solver using second-order central difference scheme
    def fps(self, nx, ny, dx, dy, f):
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
    
    # spectral cutoff filter
    def coarsen(self,nx,ny,nxc,nyc,u):        
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
    
    def bc(self,nx,ny,u):
        u[:,0] = u[:,ny]
        u[:,ny+2] = u[:,2]
        
        u[0,:] = u[nx,:]
        u[nx+2,:] = u[2,:]
        
        return u
    
    def laplacian(self,nx,ny,dx,dy,u):
        lap_bc = np.zeros((nx+3, ny+3))
        lap_bc[1:nx+2,1:ny+2] = (1.0/dx**2)*(u[2:nx+3,1:ny+2]-2.0*u[1:nx+2,1:ny+2]+u[0:nx+1,1:ny+2]) \
            + (1.0/dy**2)*(u[1:nx+2,2:ny+3]-2.0*u[1:nx+2,1:ny+2]+u[1:nx+2,0:ny+1])
           
        lap_bc = self.bc(nx,ny,lap_bc)
            
        return lap_bc
    
    # compute jacobian using arakawa scheme
    # computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
    # no ghost points
    def jacobian(self,nx,ny,dx,dy,re,w,s):
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
    
    # compute rhs using arakawa scheme
    # computed at all physical domain points (1:nx+1,1:ny+1; all boundary points included)
    # no ghost points
    def rhs(self,nx,ny,dx,dy,re,w,s):
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
    
    def energy_spectrum(self,nx,ny,dx,dy,w):
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
                        
            en[k] = en[k]/ic
            
        return en, n

    # assign random state to the environment at the begining of each episode
    def reset(self):
        
        self.time = 0.0
        self.n_start = 0
        self.n_end = self.cs_update_freq
        
        folder = 'data_'+str(int(self.re)) + '_' + str(self.nxf) + '_' + str(self.nyf)
        filename = './results/'+folder+'/results_s.npz'
        data_s = np.load(filename)
        
        sf = data_s['sall'][0,:,:]    
        wf = -self.laplacian(self.nxf,self.nyf,self.dxf,self.dyf,sf)
    
        filename = './results/'+folder+'/results_s.npz'
        wc = np.zeros((self.nx+3,self.ny+3))
        wc[1:self.nx+2,1:self.ny+2] = self.coarsen(self.nxf,self.nyf,self.nx,self.ny,wf[1:self.nxf+2,1:self.nyf+2])
        wc = self.bc(self.nx,self.ny,wc)
        
        self.w = np.copy(wc)
        self.s = self.fps(self.nx, self.ny, self.dx, self.dy, -self.w)
        self.s = self.bc(self.nx,self.ny,self.s)
        
        return self.w, self.s
    
    # Update the state of the environment
    def step(self, action):
        assert self.action_space.contains(action) , "%r (%s) invalid"%(action, type(action))
        done = False

        cs2 = action #perturbation parameters (action)
        
        aa = 1.0/3.0
        bb = 2.0/3.0
        
        for k in range(self.n_start+1,self.n_end+1):
            
            self.time = self.time + self.dt
            self.r = self.rhs(self.nx,self.ny,self.dx,self.dy,self.re,self.w,self.s)
            
            #stage-1
            self.t[1:self.nx+2,1:self.ny+2] = w[1:self.nx+2,1:self.ny+2] + \
                                              self.dt*self.r[1:self.nx+2,1:self.ny+2]
            
            self.t = self.bc(self.nx,self.ny,self.t)
            
            self.s = self.fps(self.nx, self.ny, self.dx, self.dy, -self.t)
            self.s = self.bc(self.nx,self.ny,self.s)
            
            self.r = self.rhs(self.nx,self.ny,self.dx,self.dy,self.re,self.t,self.s)
            
            #stage-2
            self.t[1:self.nx+2,1:self.ny+2] = 0.75*self.w[1:self.nx+2,1:self.ny+2] + \
                                              0.25*self.t[1:self.nx+2,1:self.ny+2] + \
                                              0.25*self.dt*self.r[1:self.nx+2,1:self.ny+2]
            
            self.t = self.bc(self.nx,self.ny,self.t)
            
            self.s = self.fps(self.nx, self.ny, self.dx, self.dy, -self.t)
            self.s = self.bc(self.nx,self.ny,self.s)
            
            self.r = self.rhs(self.nx,self.ny,self.dx,self.dy,self.re,self.t,self.s)
            
            #stage-3
            self.w[1:self.nx+2,1:self.ny+2] = aa*self.w[1:self.nx+2,1:self.ny+2] + \
                                              bb*self.t[1:self.nx+2,1:self.ny+2] + \
                                              bb*self.dt*self.r[1:self.nx+2,1:self.ny+2]
            
            self.w = self.bc(self.nx,self.ny,self.w)
            
            self.s = self.fps(self.nx,self.ny,self.dx,self.dy,-self.w)
            self.s = self.bc(self.nx,self.ny,self.s)
        
        wprobe_coarse = self.w[1:self.nx+2,1:self.ny+2][self.xprobe, self.yprobe]
#            if (k%self.freq == 0):
#                print(k, " ", self.time, ' ', np.max(self.w))
        
        reward = np.linalg.norm(wprobe_coarse - self.wprobe_fine[k,:,:])
        
        self.n_start += self.cs_update_freq
        self.n_end += self.cs_update_freq
        
        print(self.n_start, reward)
        if self.n_start == self.nt:
            done = True
        
        return self.w, self.s, reward, done, {}
    
    #function for rendering instantaneous figures, animations, etc. (retained to keep the code OpenAI Gym optimal)
    def render(self, mode='human'):
        self.n +=0

    # close the training session
    def close(self):
        return 0

#%% 
#-----------------------------------------------------------------------------#
# test the class step function
#-----------------------------------------------------------------------------#

if __name__ == '__main__':
        
    # create the instance of a class    
    env_config = {}
    env_config['n'] = 1 # write interval for states to be computed 
    
    lc = les_closure(env_config)   
    w,s = lc.reset() 
    
    action = np.array([0.2])
    
    done = False
    start = time.time()
    
    while not done:
        wf,sf,reward,done,dict_empty = lc.step(action) 
    
    print('CPU Time = ', time.time() - start)
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
    
    folder = 'data_'+str(int(re)) + '_' + str(nxc) + '_' + str(nyc)
    filename = './results/'+folder+'/plotting.npz'
    data_coarse = np.load(filename)
    
    kc = data_coarse['k']
    en0c = data_coarse['en0']
    en2c = data_coarse['en2']
    en4c = data_coarse['en4']

    en, n = lc.energy_spectrum(nxc,nyc,dxc,dyc,wf)
    
    aa = en - en4c

        