import vizdoom as vzd
import skimage.transform
import numpy as np

from gym.spaces import Discrete, Box

class ViZDoomENV:
    def __init__(self, seed, render=False, use_depth=False, use_rgb=True, reward_scale=1, frame_repeat=1, reward_reshape=False):
        # assign observation space
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        channel_num = 0
        if use_depth:
            channel_num = channel_num + 1
        if use_rgb:
            channel_num = channel_num + 3
        
        self.observation_space = Box(low=0, high=255, shape=(channel_num, 84, 84))

        self.reward_reshape = reward_reshape
        self.reward_scale = reward_scale
        
        
        game = vzd.DoomGame()

        game.load_config("ViZDoom_map/my_way_home.cfg")
        
        # game input setup
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        if use_depth:
            game.set_depth_buffer_enabled(True)
        
        
        # Adds buttons that will be allowed.
        num_buttons = game.get_available_buttons_size()
        self.action_space = Discrete(num_buttons)
        actions = [([False] * num_buttons)for i in range(num_buttons)]
        for i in range(num_buttons):
            actions[i][i] = True
        self.actions = actions
        # set frame repeat for taking action
        self.frame_repeat = frame_repeat
        
        game.set_seed(seed)
        game.set_window_visible(render)
        game.init()
        
        self.game = game
                
        
    def get_current_input(self):
        state = self.game.get_state()
        
        n = state.number
        
        if self.use_rgb:
            screen_buf = state.screen_buffer
            screen_buf = skimage.transform.resize(screen_buf, self.observation_space.shape)
            res = screen_buf
        if self.use_depth:
            depth_buf = state.depth_buffer
            depth_buf = skimage.transform.resize(depth_buf, resolution)
            depth_buf = depth_buf[np.newaxis,:]
            res = depth_buf

        if self.use_depth and self.use_rgb:
            res = np.vstack((screen_buf, depth_buf))
        
        self.last_input = (res, n)
        
        return res, n
    
    def step(self, action):
        info = {}
        
        reward = self.game.make_action(self.actions[action], self.frame_repeat)
        if self.reward_reshape:
            fixed_shaping_reward = self.game.get_game_variable(vzd.GameVariable.USER1) 
            shaping_reward = vzd.doom_fixed_to_double(fixed_shaping_reward) 
            shaping_reward = shaping_reward - self.last_total_shaping_reward
            self.last_total_shaping_reward += shaping_reward
            reward = shaping_reward
        
        done = self.game.is_episode_finished()
        if done:
            ob, n = self.last_input
            info['Episode_Total_Reward'] = self.total_reward
            info['Episode_Total_Len'] = n
        else:
            ob, n = self.get_current_input()
        
        reward = reward * self.reward_scale
        self.total_reward += reward


        return ob, reward, done, info
    
    def reset(self):
        self.last_input = None
        self.game.new_episode()
        self.last_total_shaping_reward = 0
        self.total_reward = 0
        ob, n = self.get_current_input()
        return ob
    
    def close(self):
        self.game.close()
