import os
import time


result = 'Performing iteration: 9\nIteration time: 252.143874168396\ncustom_metrics: {}\ndate: 2020-05-07_21-31-08\ndone: false\nepisode_len_mean: 200.0\nepisode_reward_max: 200.0\nepisode_reward_mean: 200.0\nepisode_reward_min: 200.0\nepisodes_this_iter: 64\nepisodes_total: 1357\nexperiment_id: 967f846aa36649819fa367866b1507b2\nhostname: nid03995\ninfo:\n  grad_time_ms: 226057.978\n  learner:\n    default_policy:\n      cur_kl_coeff: 0.02812499925494194\n      cur_lr: 9.999999747378752e-05\n      entropy: 0.47078368067741394\n      entropy_coeff: 0.0\n      kl: 0.004528602585196495\n      policy_loss: -0.0013431557454168797\n      total_loss: 318.4728088378906\n      vf_explained_var: 0.4732179641723633\n      vf_loss: 318.4739990234375\n  load_time_ms: 136.18\n  num_steps_sampled: 128000\n  num_steps_trained: 128000\n  sample_time_ms: 24114.808\n  update_time_ms: 4846.222\niterations_since_restore: 10\nnode_ip: 10.236.16.140\nnum_healthy_workers: 64\noff_policy_estimator: {}\nperf:\n  cpu_util_percent: 0.3999999999999999\n  ram_util_percent: 2.899999999999999\npid: 51846\npolicy_reward_max: {}\npolicy_reward_mean: {}\npolicy_reward_min: {}\nsampler_perf:\n  mean_env_wait_ms: 3.893005125697735\n  mean_inference_ms: 74.82609552428544\n  mean_processing_ms: 10.6613981344018\ntime_since_restore: 2580.3326478004456\ntime_this_iter_s: 251.33940148353577\ntime_total_s: 2580.3326478004456\ntimestamp: 1588887068\ntimesteps_since_restore: 128000\ntimesteps_this_iter: 12800\ntimesteps_total: 128000\ntraining_iteration: 10\n'

result_bin = result.encode('utf-8')

start_time = time.time()
with open('op.txt','wb',0) as f:
    for i in range(10):
        f.write(result_bin)
        f.flush()
        os.fsync(f)
f.close()
binary_time = time.time()-start_time
print('Binary IO time without buffering',binary_time)


start_time = time.time()
with open('op.txt','wb') as f:
    for i in range(10):
        f.write(result_bin)
        f.flush()
        os.fsync(f)
f.close()
binary_time = time.time()-start_time
print('Binary IO time with auto buffering',binary_time)

start_time = time.time()
with open('op.txt','w') as f:
    for i in range(10):
        f.write(result)
        f.flush()
        os.fsync(f)
f.close()
text_time = time.time()-start_time
print('Text IO time with auto buffering',text_time)

start_time = time.time()
with open('op.txt','wb') as f:
    for i in range(10):
        f.write(result.encode('utf-8'))
        f.flush()
        os.fsync(f)
f.close()
binary_time = time.time()-start_time
print('Binary IO with encode within print time with auto buffering',binary_time)

