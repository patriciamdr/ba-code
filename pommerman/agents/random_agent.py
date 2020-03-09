'''An agent that preforms a random action each step'''
import random

from . import BaseAgent


class RandomAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        return action_space.sample()


class RandomAgentWithoutBomb(BaseAgent):

    def act(self, obs, action_space):
        return random.randrange(5)
