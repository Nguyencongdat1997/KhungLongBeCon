import random

from players.base_player import BasePlayer


class RandomPlayer(BasePlayer):
    def __init__(self):
        super(RandomPlayer, self).__init__()
        self.type = 'Random'

    def play(self, state):
        '''
            Randomly pick an action
        :param
            state: current game state, in form of numpy.array of gray image
        :return:
            one of 3 action:
                stay - doing nothing
                jump
                bow
        '''
        possible_actions = [0,1,2]

        chosen_action = random.choice(possible_actions)
        return chosen_action