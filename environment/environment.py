import time
import numpy as np


from environment.screen_controller import ScreenController
from environment import env_config
from players.base_player import BasePlayer


class Environment:
    def __init__(self):
        self.screen_controller = ScreenController()
        self.previous_state = None

    def step(self, action):
        self.make_action(action)
        state = self.screen_controller.screen()
        done = self.check_end_game(state)
        reward = 1
        self.previous_state = state
        time.sleep(0.1)
        return reward, state, done

    def check_end_game(self, state):
        if self.previous_state is not None:
            if np.square(np.subtract(self.previous_state, state)).mean() < 0.005: # screen not changing -> game ended
                time.sleep(1)
                return True
        return False

    def make_action(self, action):
        if action == env_config.ACTIONS['jump']:
            self.screen_controller.jump()
        elif action == env_config.ACTIONS['bow']:
            self.screen_controller.bow()
        else:
            return

    def reset(self):
        self.screen_controller.restart()

    def close(self):
        pass



