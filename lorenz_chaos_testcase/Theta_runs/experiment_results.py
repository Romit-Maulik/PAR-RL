"""
Created on Wed Jun  3 10:18:29 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt


font = {'family' : 'Times New Roman',
        'size'   : 12}    
plt.rc('font', **font)

mean_reward_progress = np.zeros(shape=(300,1))
max_reward_progress = np.zeros(shape=(300,1))
min_reward_progress = np.zeros(shape=(300,1))

log_file = 'Training_iterations_400.txt'
        
with open(log_file,'r') as f:
    lines = f.readlines()
f.close()

mean_iter_num = 0
max_iter_num = 0
min_iter_num = 0
for line in lines:
    if 'episode_reward_mean' in line:
        if line.split(': ')[1] != '.nan\n':
            mean_reward_progress[mean_iter_num,0] = float(line.split(': ')[1])
            mean_iter_num = mean_iter_num + 1
    if 'episode_reward_max' in line:
        if line.split(': ')[1] != '.nan\n':
            max_reward_progress[max_iter_num,0] = float(line.split(': ')[1])
            max_iter_num = max_iter_num + 1
    if 'episode_reward_min' in line:
        if line.split(': ')[1] != '.nan\n':
            min_reward_progress[min_iter_num,0] = float(line.split(': ')[1])
            min_iter_num = min_iter_num + 1
            

#%%
fig,ax = plt.subplots(figsize=(6,3))            
plt.plot(mean_reward_progress[:250],label='Mean reward')
#plt.plot(max_reward_progress[:34],label='Max')
#plt.plot(min_reward_progress[:34],label='Min')
plt.legend()
plt.show()
fig.tight_layout()
fig.savefig('mean_reward.png',dpi=300)