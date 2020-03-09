from . import BaseAgent


class BaseDQNAgent(BaseAgent):

    def act(self, obs, action_space):
        raise NotImplementedError()
