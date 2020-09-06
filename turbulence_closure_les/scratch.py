"""
Created on Sat Sep  5 16:56:47 2020

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt 

nx = 256
ny = 256

x = np.linspace(0.0,2.0*np.pi,nx+1)
y = np.linspace(0.0,2.0*np.pi,ny+1)
x, y = np.meshgrid(x, y, indexing='ij')

npx = 4
npy = 4
shift = 0.5

xp = np.arange(0,npx)*int(nx/npx) + int(shift*nx/npx)
yp = np.arange(0,npy)*int(ny/npy) + int(shift*ny/npy)
xprobe, yprobe = np.meshgrid(xp,yp)

nxc = 32
nyc = 32

xc = np.linspace(0.0,2.0*np.pi,nxc+1)
yc = np.linspace(0.0,2.0*np.pi,nyc+1)
xc, yc = np.meshgrid(xc, yc, indexing='ij')

npx = 4
npy = 4

xpc = np.arange(0,npx)*int(nxc/npx) + int(shift*nxc/npx)
ypc = np.arange(0,npy)*int(nyc/npy) + int(shift*nyc/npy)
xprobec, yprobec = np.meshgrid(xpc,ypc)

print(x[xprobe,yprobe])
print(xc[xprobec,yprobec])

#%%
plt.plot(x[xprobe,yprobe],y[xprobe,yprobe], marker='.', color='k', linestyle='none')
plt.plot(xc[xprobec,yprobec],yc[xprobec,yprobec], marker='.', color='r', linestyle='none')