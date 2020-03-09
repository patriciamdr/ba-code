import copy

import neptune
import numpy as np
import pommerman

import torch
from torch import nn

from pommerman.dqn import utils, utility
from pommerman import constants
from pommerman.dqn.experience_replay import Memory, RecurrentMemory
from pommerman.dqn.nets.vdn import VDNMixer, CommVDNMixer


class CommArena:
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

        self.comm_mixer = CommVDNMixer().to(self.device)
        self.comm_target_mixer = copy.deepcopy(self.mixer)

        self.index_agent_dict = dict(zip([0, 2], dqn_agents))

        if resumed_optimizers is not None:
            self.optimizer = resumed_optimizers[0]
            self.comm_optimizer = resumed_optimizers[1]
        else:
            self.params = []
            self.comm_params = []
            self.add_agent_model_params()
            self.optimizer = torch.optim.Adam(self.params)
            self.comm_optimizer = torch.optim.Adam(self.comm_params)

        self.MSE_loss = nn.MSELoss()

        self.r_d_batch = None
        self.time_step = time_step

    def add_agent_model_params(self):
        for _, agent in self.index_agent_dict.items():
            params, comm_params = agent.get_model_params()
            self.params += params
            self.comm_params += comm_params

    def reset_agents(self):
        for _, agent in self.index_agent_dict.items():
            agent.reset_game_state()

    def update(self, q_s_tot, q_n_s_tot, batch, comm_update):
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
        if comm_update:
            target_values = reward_batches[:, self.time_step - 1].view(32, 1).expand(32, 2) + (
                    1 - done_batches[:, self.time_step - 1].view(32, 1).expand(32, 2)) * self.discount * q_n_s_tot
            loss = self.MSE_loss(q_s_tot, target_values)
            neptune.send_metric('comm-loss', loss)
        else:
            target_values = reward_batches[:, self.time_step - 1] + (
                    1 - done_batches[:, self.time_step - 1]) * self.discount * q_n_s_tot
            loss = self.MSE_loss(q_s_tot, target_values)
            neptune.send_metric('act-loss', loss)

        del reward_batches
        del done_batches
        del q_n_s_tot

        self.optimize(loss, comm_update)

    def optimize(self, loss, comm_update):
        if comm_update:
            optimizer = self.comm_optimizer
        else:
            optimizer = self.optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss


        self.update_target(comm_update)
        for _, agent in self.index_agent_dict.items():
            agent.update_target(comm_update)

    def update_epsilon(self):
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
            for _, agent in self.index_agent_dict.items():
                agent.update_epsilon(self.epsilon)

    def run_episode(self, opponent):
        pass

    def act_with_comm(self, actions):
        return [[action] + [0, 0] if type(action) == int else action for action in actions]

    def update_target(self, comm):
        if comm:
            target = self.comm_target_mixer
            mixer = self.comm_mixer
        else:
            target = self.target_mixer
            mixer = self.mixer

        # target network soft update
        for target_param, param in zip(target.parameters(), mixer.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


class CommArenaNOXP(CommArena):
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
            actions = self.act_with_comm(actions)
            for index, agent in self.index_agent_dict.items():
                s_a_n_s_experience = utils.get_s_a_n_s_experience(state, actions, next_state, index)
                agent.local_memory.push(s_a_n_s_experience)

            state = next_state

        self.r_d_batch = [self.local_memory.buffer]
        self.time_step = len(self.local_memory.buffer)

        # Store every agent's q-value of selected action/message in current state
        q_s_a_list = []
        q_s_m_list = []
        # Store every agent's q-value of taking best predicted action/message in next state
        q_n_s_a_list = []
        q_n_s_m_list = []

        for _, agent in self.index_agent_dict.items():
            q_a, q_m = agent.compute_q_values()
            # List of tensors with shape [1], num_tensors in list = num_dqn_agents
            q_s_a_list.append(q_a[0])
            q_n_s_a_list.append(q_a[1])
            q_s_m_list.append(q_m[0])
            q_n_s_m_list.append(q_m[1])
            del q_a
            del q_m

        # VDN
        # Stack list of tensors to get tensor of shape [num_dqn_agents, 1]/[num_dqn_agents, 1, 2]:
        # Q-values of batch per agent
        # Sum over q-values of agents per batch: output [[1, 1]]
        q_s_a_tot = self.mixer.forward(torch.stack(q_s_a_list))[0]
        q_n_s_a_tot = self.target_mixer.forward(torch.stack(q_n_s_a_list))[0]
        q_s_m_tot = self.comm_mixer.forward(torch.stack(q_s_m_list))[0]
        q_n_s_m_tot = self.comm_target_mixer.forward(torch.stack(q_n_s_m_list))[0]
        del q_s_a_list
        del q_s_m_list
        del q_n_s_a_list
        del q_n_s_m_list

        self.update(q_s_tot=q_s_a_tot, q_n_s_tot=q_n_s_a_tot, batch=self.r_d_batch, comm_update=False)
        self.update(q_s_tot=q_s_m_tot, q_n_s_tot=q_n_s_m_tot, batch=self.r_d_batch, comm_update=True)
        del q_s_a_tot
        del q_n_s_a_tot
        del q_s_m_tot
        del q_n_s_m_tot

        for _, agent in self.index_agent_dict.items():
            agent.local_memory.clear()
        self.local_memory.clear()

        self.update_epsilon()
        env.close()
        self.reset_agents()
        return episode_reward


class CommArenaXP(CommArena):

    def __init__(self, batch_size, buffer_size=5000, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.replay_buffer = RecurrentMemory(max_size=buffer_size)

    def sample(self):
        s_a_n_s_batch, r_d_batch = self.replay_buffer.sample_recurrent(self.batch_size, self.time_step)
        return s_a_n_s_batch, r_d_batch

    def calculate_mut_info(self, co_occurrence):
        p_am = co_occurrence / np.sum(co_occurrence)
        p_a = np.sum(co_occurrence, axis=0) / np.sum(co_occurrence)
        p_m = np.sum(co_occurrence, axis=1) / np.sum(co_occurrence)
        mut_info = 0
        for a in constants.Action:
            for m in range(0, constants.RADIO_VOCAB_SIZE + 1):
                if p_am[m, a.value] > 0:
                    mut_info += p_am[m, a.value] * np.log(p_am[m, a.value] / (p_a[a.value] * p_m[m]))
        return mut_info

    def ic_count_co_occurrence(self, co_occurrence, actions):
        indices = list(self.index_agent_dict.keys())
        action_1 = actions[indices[0]]
        action_2 = actions[indices[1]]

        if (type(action_1) in [tuple, list]) and (type(action_2) in [tuple, list]):
            act_1 = action_1[0]
            comm_2 = action_2[1:3]
            for m_2 in comm_2:
                co_occurrence[m_2, act_1] += 1

            act_2 = action_2[0]
            comm_1 = action_1[1:3]
            for m_1 in comm_1:
                co_occurrence[m_1, act_2] += 1

        return co_occurrence

    def sc_count_co_occurrence(self, co_occurrence, actions):
        for index, _ in self.index_agent_dict.items():
            action = actions[index]
            if type(action) in [tuple, list]:
                act = action[0]
                comm = action[1:3]
                for m in comm:
                    co_occurrence[m, act] += 1
        return co_occurrence

    def run_episode(self, opponent):
        # Create a set of agents (exactly four)
        agent_list = utility.create_agents(self.index_agent_dict, opponent)

        # Make the "Pommerman" environment using the agent list
        env = pommerman.make('PommeRadioCompetition-v2', agent_list)

        state = env.reset()
        episode_reward = []
        ic_co_occurrence = np.zeros([constants.RADIO_VOCAB_SIZE + 1, len(constants.Action)])
        sc_co_occurrence = np.zeros([constants.RADIO_VOCAB_SIZE + 1, len(constants.Action)])
        done = False
        buffer_can_sample = False
        while not done:
            # env.render()
            actions = env.act(state)

            ic_co_occurrence = self.ic_count_co_occurrence(ic_co_occurrence, actions)
            sc_co_occurrence = self.sc_count_co_occurrence(sc_co_occurrence, actions)

            next_state, reward, done, info = env.step(actions)
            if self.reward_shaping:
                reward = utility.get_shaped_reward(reward)

            episode_reward = reward

            # Store every agent's q-value of selected action/message in current state
            q_s_a_list = []
            q_s_m_list = []
            # Store every agent's q-value of taking best predicted action/message in next state
            q_n_s_a_list = []
            q_n_s_m_list = []

            actions = self.act_with_comm(actions)
            experience = (state, actions, reward, next_state, done)
            self.replay_buffer.local_memory.push(experience)

            if len(self.replay_buffer) > self.batch_size:
                buffer_can_sample = True
                s_a_n_s_batch, r_d_batch = self.sample()
                self.r_d_batch = utils.get_arena_batch(r_d_batch, list(self.index_agent_dict.keys()))

                for index, agent in self.index_agent_dict.items():
                    q_a, q_m = agent.compute_q_values(utils.get_agent_batch(s_a_n_s_batch, index))
                    # List of tensors with shape [batch_size], num_tensors in list = num_dqn_agents
                    q_s_a_list.append(q_a[0])
                    q_n_s_a_list.append(q_a[1])
                    q_s_m_list.append(q_m[0])
                    q_n_s_m_list.append(q_m[1])
                    del q_a
                    del q_m

            # VDN
            if buffer_can_sample:
                # Stack list of tensors to get tensor of shape [num_dqn_agents, batch_size]:
                # Q-values of batches per agent
                # Sum over q-values of agents per batch: output [[1, batch_size]]
                q_s_a_tot = self.mixer.forward(torch.stack(q_s_a_list))[0]
                q_n_s_a_tot = self.target_mixer.forward(torch.stack(q_n_s_a_list))[0]
                q_s_m_tot = self.comm_mixer.forward(torch.stack(q_s_m_list))[0]
                q_n_s_m_tot = self.comm_target_mixer.forward(torch.stack(q_n_s_m_list))[0]
                del q_s_a_list
                del q_s_m_list
                del q_n_s_a_list
                del q_n_s_m_list

                self.update(q_s_tot=q_s_a_tot, q_n_s_tot=q_n_s_a_tot, batch=self.r_d_batch, comm_update=False)
                self.update(q_s_tot=q_s_m_tot, q_n_s_tot=q_n_s_m_tot, batch=self.r_d_batch, comm_update=True)
                del q_s_a_tot
                del q_n_s_a_tot
                del q_s_m_tot
                del q_n_s_m_tot
            state = next_state

        ic = self.calculate_mut_info(ic_co_occurrence)
        sc = self.calculate_mut_info(sc_co_occurrence)

        self.replay_buffer.add_mem_to_buffer()

        self.update_epsilon()
        env.close()
        self.reset_agents()
        return episode_reward, sc, ic, self.optimizer.state_dict(), self.comm_optimizer.state_dict()
