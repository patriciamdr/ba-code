'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent, RandomAgentWithoutBomb
from .simple_agent import SimpleAgent
from .tensorforce_agent import TensorForceAgent
from .base_dqn_agent import BaseDQNAgent
from .dqn_agent import DQNAgent, DQNAgentNOXP, DQNAgentXP
from .dqn_agent_with_comm import CommDQNAgent, CommDQNAgentXP, CommDQNAgentNOXP
