import copy

import neptune
import numpy as np
import pommerman

import torch
from torch import nn

from pommerman.dqn import utils, utility

from pommerman.dqn.experience_replay import Memory, RecurrentMemory
from pommerman.dqn.nets.vdn import VDNMixer


class Arena:
    """Parent abstract Arena."""

    def __init__(self, device, tau, epsilon, epsilon_decay, final_epsilon, dqn_agents,
                 discount, reward_shaping=0, resumed_optimizers=None, time_step=0):
        self.device = device
        self.tau = tau
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.reward_shaping = reward_shaping

        self.mixer = VDNMixer().to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.index_agent_dict = dict(zip([0, 2], dqn_agents))

        if resumed_optimizers is not None:
            self.optimizer = resumed_optimizers[0]
        else:
            self.params = []
            self.add_agent_model_params()
            self.optimizer = torch.optim.Adam(self.params)

        self.MSE_loss = nn.MSELoss()

        self.r_d_batch = None
        self.time_step = time_step

    def add_agent_model_params(self):
        for _, agent in self.index_agent_dict.items():
            self.params += agent.get_model_params()

    def reset_agents(self):
        for _, agent in self.index_agent_dict.items():
            agent.reset_game_state()

    def update(self, q_s_a_tot, q_n_s_a_tot, batch):
        # batch: [[(r, d), (r, d)], [(r, d), (r, d)]]
        reward_batches, done_batches = [], []

        # episode: [(r, d), (r, d)]
        for episode in batch:
            reward_batch, done_batch = [], []
            # experience: (r, d)
            for experience in episode:
                # sub_batch per: [r, r]
                reward_batch.append(experience[0])
                done_batch.append(experience[1])
            # batches: [[r, r], [r, r]]
            reward_batches.append(reward_batch)
            done_batches.append(done_batch)

        reward_batches = torch.FloatTensor(np.array(reward_batches)).to(self.device)
        done_batches = torch.LongTensor(np.array(done_batches)).to(self.device)

        # Add future reward only if state not terminal
        # Target
        target_values = reward_batches[:, self.time_step - 1] + (
                1 - done_batches[:, self.time_step - 1]) * self.discount * q_n_s_a_tot

        del reward_batches
        del done_batches
        del q_n_s_a_tot

        loss = self.MSE_loss(q_s_a_tot, target_values)
        neptune.send_metric('loss', loss)
        self.optimize(loss)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del loss

        self.update_target()
        for _, agent in self.index_agent_dict.items():
            agent.update_target()

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
            for _, agent in self.index_agent_dict.items():
                agent.update_epsilon(self.epsilon)

    def run_episode(self, opponent):
        pass

    def update_target(self):
        # target network soft update
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


class ArenaNOXP(Arena):
    def __init__(self, buffer_size=10000, **kwargs):
        super().__init__(**kwargs)
        self.local_memory = Memory(max_size=buffer_size)

    def run_episode(self, opponent):
        # Create a set of agents (exactly four)
        agent_list = utility.create_agents(self.index_agent_dict, opponent)

        # Make the "Pommerman" environment using the agent list
        env = pommerman.make('PommeRadioCompetition-v2', agent_list)
        state = env.reset()

        episode_reward = []
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            next_state, reward, done, info = env.step(actions)
            if self.reward_shaping:
                reward = utility.get_shaped_reward(reward)

            episode_reward = reward

            r_d_experience = utils.get_r_d_experience(reward, done, list(self.index_agent_dict.keys()))
            self.local_memory.push(r_d_experience)
            for index, agent in self.index_agent_dict.items():
                s_a_n_s_experience = utils.get_s_a_n_s_experience(state, actions, next_state, index)
                agent.local_memory.push(s_a_n_s_experience)

            state = next_state

        self.r_d_batch = [self.local_memory.buffer]
        self.time_step = len(self.local_memory.buffer)

        # Store every agent's q-value of selected action in current state
        q_s_a_list = []
        # Store every agent's q-value of taking best predicted action in next state
        q_n_s_a_list = []

        q_s_a, q_n_s_a = None, None
        for _, agent in self.index_agent_dict.items():
            q_s_a, q_n_s_a = agent.compute_q_values()
            # List of tensors with shape [1], num_tensors in list = num_dqn_agents
            q_s_a_list.append(q_s_a)
            q_n_s_a_list.append(q_n_s_a)
        del q_s_a
        del q_n_s_a

        # VDN
        # Stack list of tensors to get tensor of shape [num_dqn_agents, 1]:
        # Q-values of batch per agent
        # Sum over q-values of agents per batch: output [[1, 1]]
        q_s_a_tot = self.mixer.forward(torch.stack(q_s_a_list))[0]
        q_n_s_a_tot = self.target_mixer.forward(torch.stack(q_n_s_a_list))[0]
        del q_s_a_list
        del q_n_s_a_list

        self.update(q_s_a_tot=q_s_a_tot, q_n_s_a_tot=q_n_s_a_tot, batch=self.r_d_batch)
        del q_s_a_tot
        del q_n_s_a_tot

        for _, agent in self.index_agent_dict.items():
            agent.local_memory.clear()
        self.local_memory.clear()

        self.update_epsilon()
        env.close()
        self.reset_agents()
        return episode_reward


class ArenaXP(Arena):

    def __init__(self, batch_size, buffer_size=5000, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.replay_buffer = RecurrentMemory(max_size=buffer_size)

    def sample(self):
        s_a_n_s_batch, r_d_batch = self.replay_buffer.sample_recurrent(self.batch_size, self.time_step)
        return s_a_n_s_batch, r_d_batch

    def run_episode(self, opponent):
        # Create a set of agents (exactly four)
        agent_list = utility.create_agents(self.index_agent_dict, opponent)

        # Make the "Pommerman" environment using the agent list
        env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        state = env.reset()
        episode_reward = []
        done = False
        buffer_can_sample = False
        while not done:
            # env.render()
            actions = env.act(state)
            next_state, reward, done, info = env.step(actions)
            if self.reward_shaping:
                reward = utility.get_shaped_reward(reward)

            episode_reward = reward

            # Store every agent's q-value of selected action in current state
            q_s_a_list = []
            # Store every agent's q-value of taking best predicted action in next state
            q_n_s_a_list = []

            experience = (state, actions, reward, next_state, done)
            self.replay_buffer.local_memory.push(experience)

            if len(self.replay_buffer) > self.batch_size:
                buffer_can_sample = True
                s_a_n_s_batch, r_d_batch = self.sample()
                self.r_d_batch = utils.get_arena_batch(r_d_batch, list(self.index_agent_dict.keys()))

                for index, agent in self.index_agent_dict.items():
                    q_s_a, q_n_s_a = agent.compute_q_values(utils.get_agent_batch(s_a_n_s_batch, index))
                    # List of tensors with shape [batch_size], num_tensors in list = num_dqn_agents
                    q_s_a_list.append(q_s_a)
                    q_n_s_a_list.append(q_n_s_a)
                    del q_s_a
                    del q_n_s_a

            # VDN
            if buffer_can_sample:
                # Stack list of tensors to get tensor of shape [num_dqn_agents, batch_size]:
                # Q-values of batches per agent
                # Sum over q-values of agents per batch: output [[1, batch_size]]
                q_s_a_tot = self.mixer.forward(torch.stack(q_s_a_list))[0]
                q_n_s_a_tot = self.target_mixer.forward(torch.stack(q_n_s_a_list))[0]
                del q_s_a_list
                del q_n_s_a_list

                self.update(q_s_a_tot=q_s_a_tot, q_n_s_a_tot=q_n_s_a_tot, batch=self.r_d_batch)
                del q_s_a_tot
                del q_n_s_a_tot
            state = next_state

        self.replay_buffer.add_mem_to_buffer()

        self.update_epsilon()
        env.close()
        self.reset_agents()
        return episode_reward, 0, 0, self.optimizer.state_dict(), None
