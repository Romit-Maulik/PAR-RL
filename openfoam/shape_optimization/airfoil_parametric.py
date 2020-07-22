"""
Created on Tue Jul 14 14:11:36 2020

@author: suraj
"""
import numpy as np
from scipy.optimize import minimize
#from simulated_annealing import sa
#from pso import pso_simple
from matplotlib import pyplot as plt

from parametric_airfoil import *

#from pyfoil.parametric_airfoil import *
#from pyfoil.xfoil import *

#%%
# lets read in the naca0012 airfoil geometry
target_airfoil = []
with open('naca0012.txt','r') as foil_txt:
    for i, row in enumerate(foil_txt.readlines()):
        if len(row.split())==2: 
            x, y = row.split()
            target_airfoil.append([float(x), float(y)])    
            
target_airfoil = np.asarray(target_airfoil)

target_airfoil_2412 = []
with open('naca2412.txt','r') as foil_txt:
    for i, row in enumerate(foil_txt.readlines()):
        if len(row.split())==2: 
            x, y = row.split()
            target_airfoil_2412.append([float(x), float(y)])    
            
target_airfoil_2412 = np.asarray(target_airfoil_2412)

#target_airfoil = target_airfoil_0012

#%%
# reverse fit the parametric airfoil to the naca0012
xpts = np.linspace(0,0.99,100) # number of xpts should be half the number of points in target airfoil for it to work
q = 3
if q == 2:
    initial_control_params = [0.03, 0.76, 0.08, 0.48, 0.13, 0.15, -0.08, 0.69, -0.03]
elif q == 3:
    initial_control_params = [0.03, 0.76, 0.08, 0.48, 0.13, 0.15, 0.12, 0.15, -0.08, 0.37, -0.01, 0.69, 0.04]

if q == 2:
    slic_x = [1,3,5,7]
    slic_y = [2,4,6,8]
elif q == 3:
    slic_x = [1,3,5,7,9,11]
    slic_y = [2,4,6,8,10,12]

px = np.array(initial_control_params)[slic_x]
py = np.array(initial_control_params)[slic_y]

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(px, py, 'o', label='Control points')
ax.plot(target_airfoil[:,0], target_airfoil[:,1], label='Base Airfoil')
ax.plot(target_airfoil_2412[:,0], target_airfoil_2412[:,1], label='2412 Airfoil')
ax.set_aspect('equal')
plt.ylim(-0.15,0.15)
plt.legend()
plt.show()


#%%
# create cost function
def shape_match(control_pts, xpts=xpts, ta=target_airfoil):
    ca = np.array(bezier_airfoil(xpts, munge_ctlpts(control_pts, q, q)))
    try:
        return 1000 * np.sum(abs(ca[:,0] - ta[:,0]) + abs(ca[:,1] - ta[:,1]))
    except:
        return 1e10

shape_match(initial_control_params)

#%%
# minimize this thing
res = minimize(shape_match, initial_control_params, method='Powell', tol=1e-10, options={'disp': True, 'maxiter':10000})

res.fun
res.x

#%%
# lets plot the solution
#res.x[0] = 0.25
matched_airfoil = np.array(bezier_airfoil(xpts, munge_ctlpts(res.x, q, q)))
x0, y0 = zip(*target_airfoil)
x1, y1 = zip(*matched_airfoil)

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x0, y0, label='target')
plt.plot(x1, y1, label='final')

ax.set_aspect('equal')
plt.ylim(-0.15,0.15)
plt.legend()
plt.show()

#%%

# the initial airfoil used panels with constant spacing
# lets be smart and change this to cosine spacing
# while using the control points we just solved for.

num_panels = 100
num_pts = num_panels + 1

x_rad = np.linspace(0, pi, num_pts)
x_cos = (np.cos(x_rad) / 2) + 0.5
x_cos = x_cos[1:]

# matched airfoil with cosine spacing
matched_airfoil_cos = np.array(bezier_airfoil(x_cos, munge_ctlpts(res.x, q, q)))

#%%
# final sanity check

x0, y0 = zip(*matched_airfoil)
x1, y1 = zip(*matched_airfoil_cos)

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x0, y0, label='equal spacing')
plt.plot(x1, y1, label='cosine spacing')

ax.set_aspect('equal')
plt.ylim(-0.15,0.15)
plt.legend()
plt.show()

#%%
# plot final airfoil with control points
x0, y0=zip(*np.array(munge_ctlpts(res.x, q, q)))
x1, y1 = zip(*matched_airfoil_cos)

px_o = np.array(res.x[slic_x])
py_o = np.array(res.x[slic_y])

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x0, y0, color='#2c3e50', linewidth=1.5, linestyle='--', label='bezier segment')
plt.plot(x1, y1, color='#2980b9', label='parametric naca0012', linewidth=2)
plt.plot(x0, y0, 'o', mfc='none', mec='r', markersize=8, label='control pts')
plt.plot(px_o, py_o, 'o', mfc='none', mec='b', markersize=10, label='control pts')

ax.set_aspect('equal')
plt.ylim(-0.15, 0.15)
plt.legend()

plt.show()

#%%
#res_x = np.array([res.x[0],res.x[1],res.x[2],res.x[5],res.x[6],res.x[7],res.x[8],res.x[11],res.x[12]])
#res_x[0] = 0.02
res_x = np.array(res.x)

res_x[0] = res_x[0] #+ 0.01
res_x[2] = res_x[2] + 0.02
res_x[4] = res_x[4] + 0.02
res_x[6] = res_x[6] + 0.016
res_x[8] = res_x[8] + 0.016
res_x[10] = res_x[10] + 0.016
res_x[12] = res_x[12] + 0.016

#res_x[8] = -0.04
#res_x[10] = -0.04 
x0, y0=zip(*np.array(munge_ctlpts(res_x, q, q)))

num_panels = 50
num_pts = num_panels + 1

x_rad = np.linspace(0, pi, num_pts)
x_cos = (np.cos(x_rad) / 2) + 0.5
x_cos = x_cos[1:]

initial_airfoil = np.array(bezier_airfoil(x_cos, munge_ctlpts(res_x, q, q)))

np.savetxt('parameterized_airfoil.csv',initial_airfoil,delimiter=",")
x1, y1 = zip(*initial_airfoil)

px_o = np.array(res_x[slic_x])
py_o = np.array(res_x[slic_y])

fig, ax = plt.subplots(figsize=(10,4))
plt.plot(x0, y0, color='#2c3e50', linewidth=1.5, linestyle='--', label='Bezier segment')
plt.plot(x1, y1, color='k', label='Parametric naca2412', linewidth=2)
plt.plot(target_airfoil_2412[:,0], target_airfoil_2412[:,1], 'g--', label='naca2412', linewidth=2)
#plt.plot(target_airfoil[:,0], target_airfoil[:,1], 'k--', label='naca0012', linewidth=2)
#plt.plot(x0, y0, 'o', mfc='none', mec='r', markersize=8, label='control pts')
plt.plot(px_o, py_o, 'o', mfc='none', mec='b', markersize=8, label='Control points')

ax.set_aspect('equal')
plt.ylim(-0.15, 0.15)
plt.legend(loc=4)

plt.show()
fig.tight_layout()
fig.savefig('parametric_naca.png',dpi=300)
