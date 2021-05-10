import time
import numpy as np


from environment.screen_controller import ScreenController
from environment import env_config
from players.base_player import BasePlayer


class Game:
    def __init__(self, player: BasePlayer):
        self.screen_controller = ScreenController()
        self.player = player

        self.game_ended = False
        self.history = []

    def run(self):
        self.screen_controller.restart()
        time.sleep(1)

        while not self.game_ended:
            self.step()
            time.sleep(0.1)

        score = 0
        return self.history, score

    def step(self):
        if self.check_end_game():
            return
        else:
            state = self.screen_controller.screen()
            action = self.player.play(state)
            self.make_action(action)
            self.history.append((state, action))
            return

    def train(self):
        pass

    def check_end_game(self):
        if len(self.history) > 2:
            if np.square(np.subtract(self.history[-1][0], self.history[-2][0])).mean() < 0.005: # screen not changing -> game ended
                self.game_ended = True
                time.sleep(1)

    def make_action(self, action):
        if action == env_config.ACTIONS['jump']:
            self.screen_controller.jump()
        elif action == env_config.ACTIONS['bow']:
            self.screen_controller.bow()
        else:
            return



