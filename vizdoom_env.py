import vizdoom as vzd
import skimage.transform
import numpy as np


# define observation and action space
class ViZDoom_observation_space:
    def __init__(self):
        self.shape = (4, 84, 84)
class Discrete:
    def __init__(self):
        self.n = 3 # number of actions
        self.shape = [self.n]

class ViZDoomENV:
    def __init__(self, seed):
        # assign observation and action space
        self.observation_space = ViZDoom_observation_space()
        self.action_space = Discrete()
        
        game = vzd.DoomGame()
        game.set_doom_scenario_path("ViZDoom_map/health_gathering.wad")
        
        # game input setup
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
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
        self.frame_repeat = 1
        
        # Causes episodes to finish after 2100 tics (actions)
        game.set_episode_timeout(2100)
        # Sets the livin reward (for each move) to 1
        reward_scale = 1e-3
        game.set_living_reward(1 * reward_scale)
        game.set_death_penalty(1000 * reward_scale)
        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.set_mode(vzd.Mode.PLAYER)

        
        
        
        game.set_seed(seed)
        game.set_window_visible(False)
        game.init()
        
        self.game = game
        
        self.last_input = None
        
        
    def get_current_input(self):
        state = self.game.get_state()
        
        n = state.number
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        
        # down sample to 84 * 84
        resolution = self.observation_space.shape[1:]
        screen_buf = skimage.transform.resize(screen_buf, resolution)
        depth_buf = skimage.transform.resize(depth_buf, resolution)
        
        # change axis
        screen_buf = np.rollaxis(screen_buf, 2, 0)
        depth_buf = depth_buf[np.newaxis,:]
        
        res = np.vstack((screen_buf, depth_buf))
        
        self.last_input = (res, n)
        
        return res, n
    
    def step(self, action):
        info = {}
        reward = self.game.make_action(self.actions[action], self.frame_repeat)
        done = self.game.is_episode_finished()
        if done:
            ob, n = self.last_input
            info['Episode_Total_Reward'] = self.game.get_total_reward()
            info['Episode_Total_Len'] = n
        else:
            ob, n = self.get_current_input()
        
        return ob, reward, done, info
    
    def reset(self):
        self.game.new_episode()
        ob, n = self.get_current_input()
        return ob
    
    def close(self):
        self.game.close()
