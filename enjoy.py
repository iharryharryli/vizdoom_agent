#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import torch

from vizdoom_env import ViZDoomENV
from model import Policy

from time import sleep

import saves as save_file_names
import argparse

import json
import storage

from collections import deque

# In[2]:


parser = argparse.ArgumentParser(description='RL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('result_dir', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--fps', type=int, default=120)
parser.add_argument('--eval', action='store_true')


# In[3]:

args = parser.parse_args()


# In[4]:


result_dir = args.result_dir


# In[5]:


model_params = os.path.join(result_dir, save_file_names.parameter_save_file)
model_params = json.load(open(model_params))

env_params = os.path.join(result_dir, save_file_names.env_parameter_save_file)
env_params = json.load(open(env_params))
if not args.eval:
    env_params['render'] = True

MODEL_SAVE_PATH = os.path.join(result_dir, save_file_names.MODEL_SAVE_PATH_file)


# In[6]:


env_params


# In[7]:


cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


# In[8]:


env = ViZDoomENV(args.seed, **env_params)


# In[9]:


recurrent_policy = True


# In[10]:


actor_critic = Policy(env.observation_space.shape, env.action_space, device,
    base_kwargs={'recurrent': recurrent_policy})
actor_critic.to(device)
actor_critic.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))


# In[11]:


recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size).to(device)
prev_action_one_hot = torch.zeros(1, env.action_space.n).to(device)

recent_count = 100
episode_rewards = deque(maxlen=recent_count)
episode_lengths = deque(maxlen=recent_count)

# In[ ]:


while True:
    obs = env.reset()
    done = False
    masks = torch.zeros(1, 1).to(device)
    
    
    while not done:
        with torch.no_grad():
            value, action, action_prob, recurrent_hidden_states = actor_critic.act(
                torch.FloatTensor(obs[np.newaxis,:]).to(device), recurrent_hidden_states, masks, 
                prev_action_one_hot, deterministic=False)
            cur_action = action
            #prev_action_one_hot = storage.actions_to_one_hot(action, env.action_space.n).to(device)

        obs, reward, done, info = env.step(action)

        masks.fill_(0.0 if done else 1.0)

        if not args.eval:
            sleep(1.0/(args.fps))

    episode_rewards.append(info['Episode_Total_Reward'])
    episode_lengths.append(info['Episode_Total_Len'])

    if args.eval:
        print("avg reward = {}, avg length = {}".format(np.mean(episode_rewards),
                                                               np.mean(episode_lengths)))
    else:
        print(info)


# In[ ]:


obs[2]


# In[ ]:


value


# In[ ]:


value, actor_features, rnn_hxs = actor_critic.base(obs, None, masks)


# In[ ]:


dist = actor_critic.dist(actor_features)


# In[ ]:


dist.sample()


# In[ ]:




