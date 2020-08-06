# Accelerating CFD solvers

In this test case, we try to check the feasibility of RL to dynamically update underrelaxation factors in CFD solvers to accelerate the convergence in CFD simulations of turbulent flows. We test our framework for the backward facing step examples where an agent is trained for different inlet velocities. The underrelaxation factors for velocity and pressure are updated after every n iterations of CFD simulation.
<p align="center">
	<img src="misc/contour_vectors.png" width="512">
</p>

## Problem formulation

The MDP problem for this test case is formulated as below
- The state of the system is the sume of average value of the square of the velocity at the inlet boundary and in the internal mesh.   
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20s_k%3D%5Cfrac%7B1%7D%7BN_b%7D%5Csum%20U_b%5E2%20&plus;%20%5Cfrac%7B1%7D%7BN_m%7D%5Csum%20U_m%5E2">
	</p>
	
- The agent chooses the underrelaxation factor for discretized momentum and pressure equation.
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20a_k%20%3D%20%5C%7B%5Calpha_u%2C%20%5Calpha_p%20%5C%7D">
	</p>
	
- The reward is the total number of iterations it took for the CFD simulations to converge
	<p align="center">
		<img src="https://latex.codecogs.com/gif.latex?%5Cdpi%7B150%7D%20r_k%20%3D%20n_%7B%5Ctext%7Biter%7D%7D">
	</p>

## Results

- We implemet proxymal policy optimization (PPO) and asynchronous proxymal policy optimization (APPO) for this test case. The PPO agent is trained for 2000 episodes and the APPO agent is trained for 3500 eisodes for different number of workers. 
<p align="center">
	<img src="misc/mean_reward_of.png" width="640">
</p>

- Once the agent is trained, it is tested for three different values of inlet velocities, V = 25.0, 50.0, 75.0 m/s. The boxplot and errorbar plot for the PPO algorithm is shown below.
<p align="center">
	<img src="misc/subplots_of_ppo.png" width="640">
</p>
The boxplot and errorbar plot for the APPO algorithm is shown below.
<p align="center">
	<img src="misc/subplots_of_appo.png" width="640">
</p>

## Running the code

## Relevant research articles
