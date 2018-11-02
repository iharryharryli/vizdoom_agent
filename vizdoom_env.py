import vizdoom as vzd
import skimage.transform
import numpy as np


# define observation and action space
class ViZDoom_observation_space:
    def __init__(self, shape):
        self.shape = shape
class Discrete:
    def __init__(self, n):
        self.n = n # number of actions
        self.shape = [self.n]

class ViZDoomENV:
    def __init__(self, seed, render=False, use_depth=True, use_rgb=True, reward_scale=1, frame_repeat=1):
        # assign observation space
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        channel_num = 0
        if use_depth:
            channel_num = channel_num + 1
        if use_rgb:
            channel_num = channel_num + 3
        self.observation_space = ViZDoom_observation_space((channel_num, 84, 84))
        
        
        game = vzd.DoomGame()
        game.set_doom_scenario_path("ViZDoom_map/health_gathering_supreme.wad")
        
        # game input setup
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        if use_depth:
            game.set_depth_buffer_enabled(True)
        
        # rendering setup
        game.set_render_hud(False)
        game.set_render_minimal_hud(False)  # If hud is enabled
        game.set_render_crosshair(False)
        game.set_render_weapon(False)
        game.set_render_decals(False)  # Bullet holes and blood on the walls
        game.set_render_particles(False)
        game.set_render_effects_sprites(False)  # Smoke and blood
        game.set_render_messages(False)  # In-game messages
        game.set_render_corpses(False)
        #game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items
        
        # Adds buttons that will be allowed.
        self.action_space = Discrete(3)
        game.add_available_button(vzd.Button.TURN_LEFT)
        game.add_available_button(vzd.Button.TURN_RIGHT)
        game.add_available_button(vzd.Button.MOVE_FORWARD)
        # generate the corresponding actions
        num_buttons = self.action_space.n
        actions = [([False] * num_buttons)for i in range(num_buttons)]
        for i in range(num_buttons):
            actions[i][i] = True
        self.actions = actions
        # set frame repeat for taking action
        self.frame_repeat = frame_repeat
        
        # Causes episodes to finish after 2100 tics (actions)
        game.set_episode_timeout(2100)
        # Sets the livin reward (for each move) to 1
        #game.set_living_reward(1 * reward_scale)
        #game.set_death_penalty(1000 * reward_scale)
        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.set_mode(vzd.Mode.PLAYER)

        
        
        
        game.set_seed(seed)
        game.set_window_visible(render)
        game.init()
        
        self.game = game
        
        
    def get_current_input(self, state):
        resolution = self.observation_space.shape[1:]
                
        if self.use_rgb:
            screen_buf = state.screen_buffer
            screen_buf = skimage.transform.resize(screen_buf, resolution)
            screen_buf = np.rollaxis(screen_buf, 2, 0)
            res = screen_buf
        if self.use_depth:
            depth_buf = state.depth_buffer
            depth_buf = skimage.transform.resize(depth_buf, resolution)
            depth_buf = depth_buf[np.newaxis,:]
            res = depth_buf

        if self.use_depth and self.use_rgb:
            res = np.vstack((screen_buf, depth_buf))
                
        return res
    
    def step(self, action):
        info = {}
        reward = 0
        
        self.game.make_action(self.actions[action], self.frame_repeat)

        done = self.game.is_episode_finished()
        if done:
            ob, n = self.last_input
            info['Episode_Total_Len'] = n
        else:
            cur_state = self.game.get_state()
            
            ob = self.get_current_input(cur_state)
            self.last_input = (ob, cur_state.number)
            current_health = cur_state.game_variables[0]
            if self.last_health is None:
                self.last_health = current_health
            else:
                reward = current_health - self.last_health
                self.last_health = current_health

        reward = reward * self.reward_scale
        
        return ob, reward, done, info
    
    def reset(self):
        self.last_input = None
        self.last_health = None
        self.game.new_episode()
        ob, n = self.get_current_input()
        return ob
    
    def close(self):
        self.game.close()
