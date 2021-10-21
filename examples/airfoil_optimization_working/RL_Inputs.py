
#save
savePath = './Case_Data'
saveInterCheckPts = True

#Global parameters
trainFlag       = True
testFlag        = False

checkpointPath  = None

numTrainEpisodes = 5000
numTestEpisodes = 3000

#Reward
reward_type  = 2 # 1: c_{l} / c_{d}; 2: c_{d}
states_type  = 1 # 1: single state; 2: k state history

#State specification for each model (inter / intra)
uinf_type = [1]   # 0: Deterministic; 1 : Stochastic
uinf_mean = [120]  # <uinf>
uinf_std  = [5]   # std(uinf); type 0 for deterministic

#MultFidelity model parameters
numModels   = 1
models      = [2] # 1: Potential Flow; 2: RANS

modelSwitch = False  # True :Multiple fidelity models; False: Same model
framework   = 'TL'  # TL : Transfer Learning (uni); MF : Multi-Fidelity (bi)

modelMaxIters       = [500] # Max iterations for each model
statWindowSize      = [500]   #Window for statistics computaion
convWindowSize      = [500]   #Window for convergence test
convCriterion       = [1]   # 0 : Immediate reward; 1: Window
varianceRedPercent  = [0.7]   #Variance reduction before switch

#Trainer parameters (for additional parameters, modity train_ppo)
trainer                      = 'PPO' # PPO / APPO / A3C
num_gpus                     = 0
num_workers                  = 4
sgd_minibatch_size           = 8
sample_batch_size            = 20
train_batch_size             = 20
vf_clip_param                = 10
metrics_smoothing_episodes   = 10






