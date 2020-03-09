# Double Deep Q-Learning Agent with fixed q targets using Dueling Recurrent CNN
import random

from pommerman import action_prune
import numpy as np
import torch
from pommerman import constants

from pommerman.dqn import utils

from pommerman.agents import BaseDQNAgent
from pommerman.dqn.experience_replay import Memory
from pommerman.dqn.nets.recurrent_cnn import PreprocessingDRCNN


class DQNAgent(BaseDQNAgent):
    """Parent abstract DQNAgent."""

    def __init__(self, device, action_model, epsilon=0, tau=0, time_step=0, batch_size=1):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.time_step = time_step

        self.device = device

        self.action_model = action_model.to(self.device)
        self.target_action_model = PreprocessingDRCNN(constants.BOARD_SIZE, len(constants.Action)).to(self.device)

        self.params = list(self.action_model.parameters())

        self.hidden_state, self.cell_state = None, None
        self.step_count = 0

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.target_action_model.parameters(), self.action_model.parameters()):
            target_param.data.copy_(param)

    def reset_game_state(self):
        super().__init__()

    def get_model_params(self):
        return self.params

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def act(self, obs, action_space):
        self.step_count = obs['step_count']
        if self.step_count == 0:
            self.hidden_state, self.cell_state = utils.init_hidden_states(batch_size=1, hidden_size=128)
            self.hidden_state = self.hidden_state.to(self.device)
            self.cell_state = self.cell_state.to(self.device)

        state = utils.preprocess_obs(obs)
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        model_out = self.action_model.forward(state, batch_size=1, time_step=1,
                                              hidden_state=self.hidden_state, cell_state=self.cell_state)
        self.hidden_state = model_out[1][0]
        self.cell_state = model_out[1][1]

        if random.uniform(0, 1) < self.epsilon:
            actions = action_prune.get_filtered_actions(obs)
            action = random.choice(actions)
        else:
            q_out = model_out[0]
            action = int(torch.argmax(q_out[0]))

        del state
        del model_out
        return action

    def compute(self, batch):
        # batch: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
        state_batches, action_batches, next_state_batches = [], [], []

        # episode: [(s, a, n_s), (s, a, n_s)]
        for episode in batch:
            state_batch, action_batch, next_state_batch = [], [], []
            # experience: (s, a, n_s)
            for experience in episode:
                # sub_batch per: [s, s]
                state_batch.append(utils.preprocess_obs(experience[0]))
                action_batch.append(experience[1])
                next_state_batch.append(utils.preprocess_obs(experience[2]))
            # batches: [[s, s], [s, s]]
            state_batches.append(state_batch)
            action_batches.append(action_batch)
            next_state_batches.append(next_state_batch)

        state_batches = torch.FloatTensor(np.array(state_batches)).to(self.device)
        action_batches = torch.LongTensor(np.array(action_batches)).to(self.device)
        next_state_batches = torch.FloatTensor(np.array(next_state_batches)).to(self.device)

        hidden_batch, cell_batch = utils.init_hidden_states(batch_size=self.batch_size, hidden_size=128)
        hidden_batch = hidden_batch.to(self.device)
        cell_batch = cell_batch.to(self.device)

        # Returns q-values of actions per batch based on complete history with output dim [batch_size, num_actions]
        q_s, _ = self.action_model.forward(state_batches, batch_size=self.batch_size,
                                           time_step=self.time_step, hidden_state=hidden_batch, cell_state=cell_batch)
        # This will give the q-value of the selected action in the last time-step per batch in array form
        # Prediction
        q_s_a = q_s.gather(dim=1, index=action_batches[:, self.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        # Double DQN
        # Use DQN to select what is the best action to take for the next state with output dim [batch_size, num_actions]
        q_next, _ = self.action_model.forward(next_state_batches, batch_size=self.batch_size,
                                              time_step=self.time_step, hidden_state=hidden_batch, cell_state=cell_batch)
        # Returns best action per batch for the next state
        _, next_action = q_next.detach().max(dim=1)
        # Use target DQN to calculate the target q-value of taking that action at the next state per batch
        q_n_s, _ = self.target_action_model.forward(next_state_batches, batch_size=self.batch_size,
                                                    time_step=self.time_step, hidden_state=hidden_batch,
                                                    cell_state=cell_batch)
        q_n_s_a = q_n_s.gather(dim=1, index=next_action.unsqueeze(dim=1)).squeeze(dim=1)

        del state_batches
        del action_batches
        del next_state_batches
        del hidden_batch
        del cell_batch
        del q_s
        del q_n_s
        del q_next

        # agent's q-values of state and next_state per batch in array form
        return q_s_a, q_n_s_a

    def compute_q_values(self, **kwargs):
        pass

    def update_target(self):
        # target network soft update
        for target_param, param in zip(self.target_action_model.parameters(), self.action_model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


class DQNAgentXP(DQNAgent):

    def compute_q_values(self, agent_batch):
        # actor update with {batch_size} episodes
        # agent_batch: list with {batch_size} episodes with {time_step} experiences
        # E.g. time_step = 2, batch_size = 2: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
        return self.compute(agent_batch)


class DQNAgentNOXP(DQNAgent):

    def __init__(self, buffer_size=10000, **kwargs):
        super().__init__(**kwargs)
        self.local_memory = Memory(max_size=buffer_size)

    def compute_q_values(self):
        # actor update with one episode of length time_step [(s, a, n_s), (s, a, n_s)]
        # batch: list with one episode [[(s, a, n_s), (s, a, n_s)]
        self.time_step = len(self.local_memory.buffer)
        return self.compute([self.local_memory.buffer])
