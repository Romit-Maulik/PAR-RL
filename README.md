# RLLib_Theta
Scaling RLLib for generic simulation environments on Theta

## Dependencies
1. Tensorflow 2.1.0
2. Gym 0.17.1
3. Ray 0.8.4
4. Numpy 1.18.1

### In progress - debugging latency on compute nodes for Ray worker start-up
### To do - funnel system calls to OpenFoam through subprocess within gym custom environment
### To do - call tensorflow surrogate models through static graph execution within gym custom environment
### To do - assess scaling of both on Theta (OpenFoam5, intel compiler)
