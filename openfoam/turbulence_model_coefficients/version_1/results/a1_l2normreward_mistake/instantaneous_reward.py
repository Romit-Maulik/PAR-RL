"""
Created on Tue Jul  7 12:19:12 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('progress_1.csv',delimiter=',')
reward = data[:,10]
slicing = reward != -100
instaneous_reward = -data[:,4][slicing]

#%%
cols = (14,8,0,1,2,3)
data = np.genfromtxt('progress.csv',delimiter=',',usecols=cols)
data = data[1:,:] 
ma_reward_max =  data[:,2]
ma_reward_min =  data[:,3]
ma_reward_mean =  data[:,4]
instaneous_episodes = data[:,1]

#%%
fig, ax  = plt.subplots(1,1,figsize=(7,4))
ax1 = ax.twinx()
ax.plot(instaneous_reward,'b',alpha=0.4,label='Instantaneous reward')
ax.plot(instaneous_episodes,ma_reward_mean,'b',lw='2',label='Moving average mean reward')
ax.plot(instaneous_episodes,ma_reward_max,'r',lw='2',label='Moving average max reward')
ax.set_ylim([-800,0])
ax1.set_ylim([-800,-0])
ax.set_xlabel('# of Episodes')
ax.set_ylabel('Reward')
ax1.set_ylabel('Reward')
ax.legend()
plt.show()
fig.tight_layout()
fig.savefig('moving_average.png',dpi=300)