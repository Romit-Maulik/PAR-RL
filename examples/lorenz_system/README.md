# Lorenz system
- **Problem formulation**
The MDP problem for the RL is formulated as below
	- The state of the system consists of the coordinates of the Lorenz sysytem and their first order derivative.   
	![](https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20s_k%3Dx%2Cy%2Cz%2C%5Cdot%7Bx%7D%2C%5Cdot%7By%7D%2C%5Cdot%7Bz%7D)
	
- **Results:**
	- The progress of training for different number of workers is shown in the Figure below. The plot shows the reward averaged over last 5 episodes (can be defined with `config[metrics_smoothing_episodes]`).  
	<p align="center">
		<img src="misc/mean_reward.png" width="512">
	</p>
	
	- The trained agent is able to successfully mantain the chaos in the Lorenz zystem. In the below Figure, the black line presents the uncontrolled trajectory and the blue line is for the controlled trajectory of the Lorenz system.       
	<p align="center">
		<img src="misc/results_summary.png" width="768">
	</p>

- **Directory layout**

- **Running the code**

- **Research Articles:**  

	[Restoring chaos using deep reinforcement learning](https://aip.scitation.org/doi/abs/10.1063/5.0002047?journalCode=cha)




