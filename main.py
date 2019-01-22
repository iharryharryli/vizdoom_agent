#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs_ViZDoom, make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
import saves as save_file_names

import os
import json



args = get_args()

env_arg = {
    "reward_scale": args.reward_scale,
    "use_rgb": not args.disable_rgb,
    "use_depth": args.use_depth,
    "frame_skip": args.frame_skip,
    "game_config": args.game_config,
    "jitter_rgb": args.jitter_rgb
}

result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)

reward_history = os.path.join(result_dir, save_file_names.reward_history_file)
loss_history = os.path.join(result_dir, save_file_names.loss_history_file)
parameter_save = os.path.join(result_dir, save_file_names.parameter_save_file)
env_parameter_save = os.path.join(result_dir, save_file_names.env_parameter_save_file)
progress_save = os.path.join(result_dir, save_file_names.progress_save_file)
MODEL_SAVE_PATH = os.path.join(result_dir, save_file_names.MODEL_SAVE_PATH_file)
fileL = [reward_history, loss_history, parameter_save, env_parameter_save]

#remove old record files
for f in fileL:
    try:
        os.remove(f)
    except OSError:
        pass


parameters = {}
parameters['algo'] = args.algo
parameters['gamma'] = args.gamma
parameters['num_steps'] = args.num_steps
parameters['num_processes'] = args.num_processes
parameters['value_loss_coef'] = args.value_loss_coef
parameters['mse_coef'] = args.mse_coef
parameters['kl_coef'] = args.kl_coef
parameters['eps'] = args.eps
parameters['entropy_coef'] = args.entropy_coef
parameters['lr'] = args.lr
parameters['use_gae'] = args.use_gae
parameters['max_grad_norm'] = args.max_grad_norm
parameters['seed'] = args.seed
parameters['recurrent_policy'] = args.recurrent_policy

if parameters['algo'] == "a2c":
    parameters['alpha'] = args.alpha
    parameters['use_adam'] = not args.use_rmsprop
elif parameters['algo'] == "ppo":
    parameters['clip_param'] = args.clip_param
    parameters['ppo_epoch'] = args.ppo_epoch
    parameters['num_mini_batch'] = args.num_mini_batch

if parameters['use_gae']:
    parameters['tau'] = args.tau
    




json.dump(parameters, open(parameter_save, "w"))
json.dump(env_arg, open(env_parameter_save, "w"))



num_updates = int(args.num_frames) // args.num_steps // args.num_processes
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

envs = make_vec_envs_ViZDoom(args.seed, args.num_processes, device, **env_arg)

actor_critic = Policy(envs.observation_space.shape, envs.action_space, device,
    base_kwargs={'recurrent': args.recurrent_policy})
actor_critic.to(device)

if args.algo == 'a2c':
    agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, args.mse_coef, args.kl_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm,
                               use_adam=parameters['use_adam'])
else:
    agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)

rollouts = RolloutStorage(args.num_steps, args.num_processes,
                    envs.observation_space.shape, envs.action_space,
                    actor_critic.recurrent_hidden_state_size)


obs = envs.reset()
rollouts.obs[0].copy_(obs)
rollouts.to(device)

recent_count = 50
episode_rewards = deque(maxlen=recent_count)
episode_lengths = deque(maxlen=recent_count)

progress = {
    "last_saved_num_updates": 0
}

for j in range(num_updates):
    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                    rollouts.prev_action_one_hot[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'Episode_Total_Reward' in info.keys():
                episode_rewards.append(info['Episode_Total_Reward'])
            if 'Episode_Total_Len' in info.keys():
                episode_lengths.append(info['Episode_Total_Len'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                   for done_ in done])
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

    
    

    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.obs[-1],
                                            rollouts.recurrent_hidden_states[-1],
                                            rollouts.masks[-1],
                                            rollouts.prev_action_one_hot[-1]).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

    losses = agent.update(rollouts)

    rollouts.after_update()
    
    total_num_steps = (j + 1) * args.num_processes * args.num_steps
    
    with open(loss_history, 'a') as the_file:
        the_file.write(("{} " * (len(losses) + 1) + "\n").format(total_num_steps, *losses))
    
    if len(episode_rewards) > 0:
        print("{} updates: avg reward = {}, avg length = {}".format(total_num_steps, np.mean(episode_rewards),
                                                               np.mean(episode_lengths)))
        
        with open(reward_history, 'a') as the_file:
            the_file.write('{} {} {} \n'.format(total_num_steps, np.mean(episode_rewards),
                                               np.mean(episode_lengths)))

    if j % args.save_interval == 0:
        torch.save(actor_critic.state_dict(), MODEL_SAVE_PATH)
        progress['last_saved_num_updates'] = j
        json.dump(progress, open(progress_save, "w"))
        
