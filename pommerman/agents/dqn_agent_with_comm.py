# Double Deep Q-Learning Agent, able to communicate, with fixed q targets using Dueling Recurrent CNN
import random

from pommerman import action_prune
from random import sample

import numpy as np
import torch
from torch import nn
from pommerman import constants

from pommerman.dqn import utils

from pommerman.agents import BaseDQNAgent
from pommerman.dqn.experience_replay import Memory
from pommerman.dqn.nets.recurrent_cnn import CommDRCNN, PreprocessingDRCNN


class CommDQNAgent(BaseDQNAgent):
    """Parent abstract DQNAgent."""

    def __init__(self, device, action_model, comm_model, epsilon=0, tau=0, time_step=0, batch_size=1):
        super().__init__()
        self.epsilon = epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.time_step = time_step

        self.device = device

        self.action_model = action_model.to(self.device)
        self.target_action_model = CommDRCNN(constants.BOARD_SIZE, len(constants.Action)).to(self.device)

        self.comm_model = comm_model.to(self.device)
        self.target_comm_model = PreprocessingDRCNN(constants.BOARD_SIZE, constants.RADIO_VOCAB_SIZE + 1).to(self.device)

        self.params = list(self.action_model.parameters())
        self.comm_params = list(self.comm_model.parameters())

        self.a_hidden_state, self.a_cell_state = None, None
        self.m_hidden_state, self.m_cell_state = None, None
        self.step_count = 0
        self.m_teammate = None

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.target_action_model.parameters(), self.action_model.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_comm_model.parameters(), self.comm_model.parameters()):
            target_param.data.copy_(param)

    def reset_game_state(self):
        super().__init__()

    def get_model_params(self):
        return self.params, self.comm_params

    def update_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def act(self, obs, action_space):
        self.step_count = obs['step_count']
        if self.step_count == 0:
            self.a_hidden_state, self.a_cell_state = utils.init_hidden_states(batch_size=1, hidden_size=128)
            self.m_hidden_state, self.m_cell_state = utils.init_hidden_states(batch_size=1, hidden_size=128)
            self.a_hidden_state = self.a_hidden_state.to(self.device)
            self.m_hidden_state = self.m_hidden_state.to(self.device)
            self.a_cell_state = self.a_cell_state.to(self.device)
            self.m_cell_state = self.m_cell_state.to(self.device)

        m_teammate = obs['message']
        m_teammate_tensor = torch.LongTensor(np.array(m_teammate)).to(self.device)
        m_teammate_tensor = nn.functional.one_hot(m_teammate_tensor, num_classes=9).float().to(self.device)
        state = utils.preprocess_obs(obs)
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)

        a_model_out = self.action_model.forward(state, teammate_message=m_teammate_tensor, batch_size=1, time_step=1,
                                                hidden_state=self.a_hidden_state, cell_state=self.a_cell_state)
        self.a_hidden_state = a_model_out[1][0]
        self.a_cell_state = a_model_out[1][1]

        if random.uniform(0, 1) < self.epsilon:
            actions = action_prune.get_filtered_actions(obs)
            action = random.choice(actions)
        else:
            a_q_out = a_model_out[0]
            action = int(torch.argmax(a_q_out[0]))

        m_model_out = self.comm_model.forward(state, batch_size=1, time_step=1,
                                              hidden_state=self.m_hidden_state, cell_state=self.m_cell_state)
        self.m_hidden_state = m_model_out[1][0]
        self.m_cell_state = m_model_out[1][1]

        assert action_space.spaces[1].n == action_space.spaces[2].n
        if random.uniform(0, 1) < self.epsilon:
            message = sample(range(1, action_space.spaces[1].n + 1), 2)
        else:
            m_q_out = m_model_out[0]
            _, message_indices = m_q_out[0][1:].topk(2, largest=True, sorted=True)
            message = [i+1 for i in message_indices.tolist()]

        del state
        del a_model_out
        del m_model_out
        return [action] + message

    def compute_a_q_values(self, batch):
        # batch: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
        state_batches, teammate_m_batches, action_batches, next_state_batches, next_teammate_m_batches = [], [], [], [], []

        # episode: [(s, a, n_s), (s, a, n_s)]
        for episode in batch:
            state_batch, teammate_m_batch, action_batch, next_state_batch, next_teammate_m_batch = [], [], [], [], []
            # experience: (s, a, n_s)
            for experience in episode:
                # sub_batch per: [s, s]
                state_batch.append(utils.preprocess_obs(experience[0]))
                teammate_m_batch.append(experience[0]['message'])
                action_batch.append(experience[1][0])
                next_state_batch.append(utils.preprocess_obs(experience[2]))
                next_teammate_m_batch.append(experience[2]['message'])
            # batches: [[s, s], [s, s]]
            state_batches.append(state_batch)
            teammate_m_batches.append(teammate_m_batch)
            action_batches.append(action_batch)
            next_state_batches.append(next_state_batch)
            next_teammate_m_batches.append(next_teammate_m_batch)

        state_batches = torch.FloatTensor(np.array(state_batches)).to(self.device)
        teammate_m_batches = torch.LongTensor(np.array(teammate_m_batches)).to(self.device)
        teammate_m_batches = nn.functional.one_hot(teammate_m_batches, num_classes=9).float().to(self.device)
        action_batches = torch.LongTensor(np.array(action_batches)).to(self.device)
        next_state_batches = torch.FloatTensor(np.array(next_state_batches)).to(self.device)
        next_teammate_m_batches = torch.LongTensor(np.array(next_teammate_m_batches)).to(self.device)
        next_teammate_m_batches = nn.functional.one_hot(next_teammate_m_batches, num_classes=9).float().to(self.device)

        hidden_batch, cell_batch = utils.init_hidden_states(batch_size=self.batch_size, hidden_size=128)
        hidden_batch = hidden_batch.to(self.device)
        cell_batch = cell_batch.to(self.device)

        # Returns q-values of actions per batch based on complete history with output dim [batch_size, num_actions]
        q_s, _ = self.action_model.forward(state_batches, teammate_message=teammate_m_batches,
                                           batch_size=self.batch_size, time_step=self.time_step,
                                           hidden_state=hidden_batch, cell_state=cell_batch)
        # This will give the q-value of the selected action in the last time-step per batch in array form
        # Prediction
        q_s_a = q_s.gather(dim=1, index=action_batches[:, self.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        # Double DQN
        # Use DQN to select what is the best action to take for the next state with output dim [batch_size, num_actions]
        q_next, _ = self.action_model.forward(next_state_batches, teammate_message=next_teammate_m_batches,
                                              batch_size=self.batch_size,
                                              time_step=self.time_step, hidden_state=hidden_batch,
                                              cell_state=cell_batch)
        # Returns best action per batch for the next state
        _, next_action = q_next.detach().max(dim=1)
        # Use target DQN to calculate the target q-value of taking that action at the next state per batch
        q_n_s, _ = self.target_action_model.forward(next_state_batches, teammate_message=next_teammate_m_batches,
                                                    batch_size=self.batch_size,
                                                    time_step=self.time_step, hidden_state=hidden_batch,
                                                    cell_state=cell_batch)
        q_n_s_a = q_n_s.gather(dim=1, index=next_action.unsqueeze(dim=1)).squeeze(dim=1)

        del state_batches
        del teammate_m_batches
        del action_batches
        del next_state_batches
        del next_teammate_m_batches
        del hidden_batch
        del cell_batch
        del q_s
        del q_n_s
        del q_next

        # agent's q-values of state and next_state per batch in array form
        return q_s_a, q_n_s_a

    def compute_m_q_values(self, batch):
        # batch: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
        state_batches, m_index_batches, next_state_batches = [], [], []

        # episode: [(s, a, n_s), (s, a, n_s)]
        for episode in batch:
            state_batch, m_index_batch, next_state_batch = [], [], []
            # experience: (s, a, n_s)
            for experience in episode:
                # sub_batch per: [s, s]
                state_batch.append(utils.preprocess_obs(experience[0]))
                m_index_batch.append(experience[1][1:3])
                next_state_batch.append(utils.preprocess_obs(experience[2]))
            # batches: [[s, s], [s, s]]
            state_batches.append(state_batch)
            m_index_batches.append(m_index_batch)
            next_state_batches.append(next_state_batch)
        state_batches = torch.FloatTensor(np.array(state_batches)).to(self.device)
        m_index_batches = torch.LongTensor(np.array(m_index_batches)).to(self.device)
        next_state_batches = torch.FloatTensor(np.array(next_state_batches)).to(self.device)

        hidden_batch, cell_batch = utils.init_hidden_states(batch_size=self.batch_size, hidden_size=128)
        hidden_batch = hidden_batch.to(self.device)
        cell_batch = cell_batch.to(self.device)

        # Returns q-values of messages per batch based on complete history with output dim [batch_size, vocab_size + 1]
        q_s, _ = self.comm_model.forward(state_batches, batch_size=self.batch_size,
                                         time_step=self.time_step, hidden_state=hidden_batch, cell_state=cell_batch)
        # This will give the q-values of the selected messages in the last time-step per batch
        # Prediction
        q_s_m = q_s.unsqueeze(1).gather(dim=2, index=m_index_batches[:, self.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        # Double DQN
        # Use DQN to select what is the best message to take for the next state with output dim [batch_size, vocab_size + 1]
        q_next, _ = self.comm_model.forward(next_state_batches, batch_size=self.batch_size,
                                            time_step=self.time_step, hidden_state=hidden_batch,
                                            cell_state=cell_batch)
        # Returns best two messages per batch for the next state
        _, next_message_indices = q_next.detach()[:, 1:].topk(2, largest=True, sorted=True)
        indices_as_tensor = torch.LongTensor(np.array([[i+1 for i in list] for list in next_message_indices.tolist()])).to(self.device)
        # Use target DQN to calculate the target q-value of sending that message at the next state per batch
        q_n_s, _ = self.target_comm_model.forward(next_state_batches, batch_size=self.batch_size,
                                                  time_step=self.time_step, hidden_state=hidden_batch,
                                                  cell_state=cell_batch)
        q_n_s_m = q_n_s.unsqueeze(1).gather(dim=2, index=indices_as_tensor.unsqueeze(dim=1)).squeeze(dim=1)

        del state_batches
        del m_index_batches
        del indices_as_tensor
        del next_state_batches
        del hidden_batch
        del cell_batch
        del q_s
        del q_n_s
        del q_next

        # agent's q-values of state and next_state per batch in array form
        return q_s_m, q_n_s_m

    def compute(self, batch):
        q_a = self.compute_a_q_values(batch)
        q_m = self.compute_m_q_values(batch)
        return q_a, q_m

    def compute_q_values(self, **kwargs):
        pass

    def update_target(self, comm_update):
        if comm_update:
            target = self.target_comm_model
            mixer = self.comm_model
        else:
            target = self.target_action_model
            mixer = self.action_model

        # target network soft update
        for target_param, param in zip(target.parameters(), mixer.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


class CommDQNAgentXP(CommDQNAgent):

    def compute_q_values(self, agent_batch):
        # actor update with {batch_size} episodes
        # agent_batch: list with {batch_size} episodes with {time_step} experiences
        # E.g. time_step = 2, batch_size = 2: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
        return self.compute(agent_batch)


class CommDQNAgentNOXP(CommDQNAgent):

    def __init__(self, buffer_size=10000, **kwargs):
        super().__init__(**kwargs)
        self.local_memory = Memory(max_size=buffer_size)

    def compute_q_values(self):
        # actor update with one episode of length time_step [(s, a, n_s), (s, a, n_s)]
        # batch: list with one episode [[(s, a, n_s), (s, a, n_s)]
        self.time_step = len(self.local_memory.buffer)
        return self.compute([self.local_memory.buffer])
