"""
Created on Mon Jun 15 18:26:01 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cols = (14,0,1,2,3)
progress_4 = np.genfromtxt('progress_4_1.csv',delimiter=',',usecols=cols)
progress_16 = np.genfromtxt('progress_4_4.csv',delimiter=',',usecols=cols)
progress_32 = np.genfromtxt('progress_4_8.csv',delimiter=',',usecols=cols)

data_4 = np.load('results_4_1.npz')
data_4c = data_4['controlled']
data_4uc = data_4['uncontrolled']

data_16 = np.load('results_4_4.npz')
data_16c = data_16['controlled']
data_16uc = data_16['uncontrolled']

data_32 = np.load('results_4_8.npz')
data_32c = data_32['controlled']
data_32uc = data_32['uncontrolled']

plt.plot(progress_4[:,0],progress_4[:,3],label='N=4')
plt.plot(progress_16[:,0],progress_16[:,3],label='N=16')
plt.plot(progress_32[:,0],progress_32[:,3],label='N=32')
plt.legend()
plt.show()

#%%
# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=(12,9))
#fig.suptitle('A tale of 2 subplots')

# First subplot
ax = fig.add_subplot(2, 2, 1)

ax.plot(progress_4[:,0],progress_4[:,3],color='blue',lw=2,label='N=4')
ax.plot(progress_16[:,0],progress_16[:,3],color='red',lw=2,label='N=16')
ax.plot(progress_32[:,0],progress_32[:,3],color='green',lw=2,label='N=32')
ax.grid()
ax.legend()

#ax.grid(True)
ax.set_ylabel('Mean reward')
ax.set_xlabel('Wall time')

# Second subplot
ax = fig.add_subplot(2, 2, 2,projection='3d')
ax.plot(data_4c[:, 1], data_4c[:, 2], data_4c[:, 3],color='steelblue',label='Controlled')
ax.plot(data_4uc[:, 1], data_4uc[:, 2], data_4uc[:, 3],color='k',label='Uncontrolled')
ax.set_title('N=4')

# Third subplot
ax = fig.add_subplot(2, 2, 3,projection='3d')
ax.plot(data_16c[:, 1], data_16c[:, 2], data_16c[:, 3],color='steelblue',label='Controlled')
ax.plot(data_16uc[:, 1], data_16uc[:, 2], data_16uc[:, 3],color='k',label='Uncontrolled')
ax.set_title('N=16')

# Third subplot
ax = fig.add_subplot(2, 2, 4,projection='3d')
ax.plot(data_32c[:, 1], data_32c[:, 2], data_32c[:, 3],color='steelblue',label='Controlled')
ax.plot(data_32uc[:, 1], data_32uc[:, 2], data_32uc[:, 3],color='k',label='Uncontrolled')
ax.set_title('N=32')

fig.tight_layout()
plt.show()
fig.savefig('results_summary.png',dpi=300)
