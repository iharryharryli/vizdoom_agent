import vizdoom as vzd
import skimage.transform
import numpy as np
import random

from gym.spaces import Discrete, Box

from skimage.util import random_noise
import scipy.ndimage

def corrupt_rgb(ob, var):
    res = random_noise(ob / 255.0, var=var) * 255.0
    return res

drop_input_init_safe_len = 10
drop_input_freq = 10

class ViZDoomENV:
    def __init__(self, seed, game_config, render=False, use_depth=False, use_rgb=True, reward_scale=1, frame_skip=4, jitter_rgb=False,
                    noise_var=0.2, drop_input_prob=0.0, rotate_sensor=False, rotate_range=30):
        # assign observation space
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        channel_num = 0
        if use_depth:
            channel_num = channel_num + 1
        if use_rgb:
            channel_num = channel_num + 3
        
        self.observation_shape = (channel_num, 84, 84)
        self.observation_space = Box(low=0, high=255, shape=self.observation_shape)

        self.reward_scale = reward_scale

        self.jitter_rgb = jitter_rgb
        self.noise_var = noise_var
        self.drop_input_prob = drop_input_prob
        self.prepare_drop_input()
        self.rotate_sensor = rotate_sensor
        self.rotate_range = rotate_range
        
        
        game = vzd.DoomGame()

        game.load_config(game_config)
        
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
        # set frame skip for taking action
        self.frame_skip = frame_skip
        
        game.set_seed(seed)
        random.seed(seed)
        game.set_window_visible(render)
        game.init()
        
        self.game = game

    def prepare_drop_input(self):
        self.is_dropping_input = False
        self.dropped_input = np.zeros(self.observation_shape)
        
    def get_current_input(self):
        if self.is_dropping_input:
            res = self.dropped_input
        else:
            state = self.game.get_state()

            res_source = []
                    
            if self.use_rgb:
                res_source.append(state.screen_buffer)
            if self.use_depth:
                res_source.append(state.depth_buffer[np.newaxis,:])

            res = np.vstack(res_source)

            # resize
            res = skimage.transform.resize(res, self.observation_space.shape, preserve_range=True)

            if self.jitter_rgb:
                res[:3] = corrupt_rgb(res[:3], self.noise_var)

            if self.rotate_sensor:
                rotate_degree = random.random() * 2 * self.rotate_range - self.rotate_range
                for i in range(self.observation_shape[0]):
                    res[i] = scipy.ndimage.rotate(res[i], rotate_degree, reshape=False)
        
        self.last_input = res
        
        return res

    def step_with_skip(self, action):
        reward_acc = 0
        ob = self.last_input

        for i in range(self.frame_skip + 1):
            reward = self.game.make_action(self.actions[action])
            reward_acc += reward
            done = self.game.is_episode_finished()

            if done:
                break
            else:
                ob = self.get_current_input()

        return ob, reward_acc, done

    
    def step(self, action):
        #decide if drop input
        if self.drop_input_prob > 0.00001:
            if self.total_length > drop_input_init_safe_len and \
            self.total_length % drop_input_freq == 0:
                self.is_dropping_input = (random.random() < self.drop_input_prob)

        ob, reward, done = self.step_with_skip(action)

        #reward scaling
        reward = reward * self.reward_scale

        self.total_reward += reward
        self.total_length += 1

        ob_is_valid = 1.0
        if self.is_dropping_input:
            ob_is_valid = 0.0
        info = {'ob_is_valid': ob_is_valid}

        if done:
            info['Episode_Total_Reward'] = self.total_reward
            info['Episode_Total_Len'] = self.total_length

        return ob, reward, done, info
    
    def reset(self):
        self.game.new_episode()
        self.total_reward = 0
        self.total_length = 0
        ob = self.get_current_input()
        return ob
    
    def close(self):
        self.game.close()
