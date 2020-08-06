# Lorenz system
In this test case, the chaos is mantained in the Lorenz system by controlling its parameters based on the magnitude of the velocity. 

## Problem formulation

The MDP problem for this test case is formulated as below
- The state of the system consists of the coordinates of the Lorenz sysytem and their first order derivative.   
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20s_k%3Dx%2Cy%2Cz%2C%5Cdot%7Bx%7D%2C%5Cdot%7By%7D%2C%5Cdot%7Bz%7D">
	</p>
	
- The agent chooses the perturbation in the parameters of the Lorenz system as action to keep the system chaotic.
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20a_k%3D%5CDelta%20%5Csigma%2C%20%5CDelta%20%5Crho%2C%20%5CDelta%20%5Cbeta">
	</p>
	
- The stepwise reward is assigned based on the magnitude of the velocity (![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20V%3D%5Csqrt%7B%28x%5E2&plus;y%5E2&plus;z%5E2%29%7D)). Additionally, the teminal reward is assigned based on the avergae reward (![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbar%7Br_t%7D)) over last 2000 time steps 
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20%5Cbegin%7Balign*%7D%20r_t%20%26%3D%20%5Cbegin%7Bcases%7D%2010%2C%20%26%5Cquad%20V%28t%29%20%3E%20V_0%2C%5C%5C%20-10.%20%26%5Cquad%20V%28t%29%20%5Cle%20V_0%2C%20%5Cend%7Bcases%7D%20%5C%5C%20r_%7Bterminal%7D%20%26%3D%20%5Cbegin%7Bcases%7D%20-100%2C%20%26%5Cquad%20%5Cbar%7Br_t%7D%20%3C%20-2%2C%5C%5C%200%2C%20%26%5Cquad%20%5Cbar%7Br_t%7D%20%3E%20-2.%20%5Cend%7Bcases%7D%20%5Cend%7Balign*%7D">
	</p>

## Results

The agent is trained for 50 episodes and each episode is divided into 4000 time steps.

- The progress of training for different number of workers is shown in the Figure below. The plot shows the reward averaged over last 5 episodes (can be defined with `config[metrics_smoothing_episodes]`).  
	<p align="center">
		<img src="misc/mean_reward.png" width="512">
	</p>
	
- The trained agent is able to successfully mantain the chaos in the Lorenz zystem. In the below Figure, the black line presents the uncontrolled trajectory and the blue line is for the controlled trajectory of the Lorenz system.       
	<p align="center">
		<img src="misc/results_summary.png" width="768">
	</p>

## Running the code
	
The training can be started by running `train_appo.py` on local machine. The number of workers can be set using the `num_workers` parameter in the config dictionary. The job can be submitted on Theta either in the `debug` or `default` mode. Job submission scripts are provided for both `debug` or `default` mode. The user has to specify the project name and RLLib environment in job submission scripts before submitting it. To submit the job in `debug` mode on Theta execute 
```
qsub ray_python_debug.sh
```

## Research Articles

[Restoring chaos using deep reinforcement learning](https://aip.scitation.org/doi/abs/10.1063/5.0002047?journalCode=cha)




