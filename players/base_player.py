from abc import ABC, abstractmethod


from environment.env_config import ACTIONS


class BasePlayer(ABC):
    def __init__(self):
        super(BasePlayer, self).__init__()

    @abstractmethod
    def play(self, state):
        """
            Try to make an action based on the observation state
        :param
            state: current game state, in form of numpy.array of gray image
        :return:
            one of 3 action:
                stay - doing nothing
                jump
                bow
        """
        return ACTIONS['stay']