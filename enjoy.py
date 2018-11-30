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


# In[2]:


parser = argparse.ArgumentParser(description='RL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('result_dir', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--fps', type=int, default=60)


# In[3]:

args = parser.parse_args()


# In[4]:


result_dir = args.result_dir


# In[5]:


model_params = os.path.join(result_dir, save_file_names.parameter_save_file)
model_params = json.load(open(model_params))

env_params = os.path.join(result_dir, save_file_names.env_parameter_save_file)
env_params = json.load(open(env_params))
env_params['render'] = True
frame_repeat = env_params['frame_skip'] + 1
env_params['frame_skip'] = 1

MODEL_SAVE_PATH = os.path.join(result_dir, save_file_names.MODEL_SAVE_PATH_file)


# In[6]:


env_params


# In[7]:


cuda = False
device = torch.device("cuda:0" if cuda else "cpu")


# In[8]:


env = ViZDoomENV(args.seed, **env_params)


# In[9]:


recurrent_policy = False
if "recurrent_policy" in model_params:
    recurrent_policy = model_params["recurrent_policy"]


# In[10]:


actor_critic = Policy(env.observation_space.shape, env.action_space,
    base_kwargs={'recurrent': recurrent_policy})
actor_critic.to(device)
actor_critic.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))


# In[11]:


recurrent_hidden_states = None
if recurrent_policy:
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)


# In[ ]:


while True:
    obs = env.reset()
    done = False
    masks = torch.zeros(1, 1)
    
    cur_action = None
    cur_action_count = 0
    
    while not done:
        if cur_action is None or cur_action_count == frame_repeat:
            cur_action_count = 0
            with torch.no_grad():
                value, action, action_prob, recurrent_hidden_states = actor_critic.act(
                    torch.FloatTensor(obs[np.newaxis,:]), recurrent_hidden_states, masks, deterministic=False)
                cur_action = action

        obs, reward, done, info = env.step(action)
        cur_action_count += 1

        masks.fill_(0.0 if done else 1.0)

        sleep(1.0/args.fps)

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




