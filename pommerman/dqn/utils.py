import os
import random

import numpy as np
import torch

from pommerman import utility, constants
from pommerman.dqn.nets.recurrent_cnn import CommDRCNN, PreprocessingDRCNN


def load_trained_team(comm, team_dir, random_agent, resume_training):
    assert team_dir is not None

    # TODO: Add handling for random agent without bomb
    if random_agent:
        prefix = '10000-RandomAgent-'
    else:
        # Adapt prefix depending on number of episodes
        prefix = '30000-Self-'

    def get_kwargs(**kwargs):
        return kwargs

    # Relevant args for agents
    agent_kwargs = get_kwargs(**{})
    device = get_device()

    # load trained team
    team = []
    optimizers = []
    if comm:
        for index in range(2):
            action_model = CommDRCNN(constants.BOARD_SIZE, len(constants.Action))
            comm_model = PreprocessingDRCNN(constants.BOARD_SIZE, constants.RADIO_VOCAB_SIZE + 1)

            checkpoint = torch.load(os.path.join(team_dir, prefix + 'agent_' + str(index) + '.tar'))
            action_model.load_state_dict(checkpoint['action_model_' + str(index)])
            comm_model.load_state_dict(checkpoint['comm_model_' + str(index)])

            if resume_training:
                action_model.train()
                comm_model.train()
            else:
                action_model.eval()
                comm_model.eval()

            agent_kwargs['action_model'] = action_model
            agent_kwargs['comm_model'] = comm_model
            team.insert(index, agent_kwargs)

        if resume_training:
            checkpoint = torch.load(os.path.join(team_dir, prefix + 'optimizers' + '.tar'))
            optimizer_dict = checkpoint['optimizer']
            comm_optimizer_dict = checkpoint['comm_optimizer']

            optimizers.insert(0, optimizer_dict)
            optimizers.insert(1, comm_optimizer_dict)

        del checkpoint
    else:
        for index in range(2):
            action_model = PreprocessingDRCNN(constants.BOARD_SIZE, len(constants.Action))

            checkpoint = torch.load(os.path.join(team_dir, prefix + 'agent_' + str(index) + '.tar'))
            action_model.load_state_dict(checkpoint['action_model_' + str(index)])

            if resume_training:
                action_model.train()
            else:
                action_model.eval()

            agent_kwargs['action_model'] = action_model
            team.insert(index, agent_kwargs)

        if resume_training:
            checkpoint = torch.load(os.path.join(team_dir, prefix + 'optimizers' + '.tar'))
            optimizer_dict = checkpoint['optimizer']

            optimizers.insert(0, optimizer_dict)

        del checkpoint
    return team, optimizers


def get_path_from_setting(_agents, result_path):
    dict_name = "_".join(_agents[:2])
    _dict = os.path.join(result_path, dict_name)
    return _dict


def randomly_select_trained_team(comm):
    # Path where trained models are located
    home_directory = os.path.expanduser('~')
    result_path = os.path.join(home_directory, 'dev/playground/models')

    if comm is True:
        path = os.path.join(result_path, 'comm_xp_results')
    else:
        path = os.path.join(result_path, 'xp_results')

    # select random saved model for each round
    dirs = [root for root, dirs, files in os.walk(path) if 'trial' in root]
    team_dir = random.choice(dirs)

    print('Randomly selected team: ' + team_dir)

    return team_dir


def bombs_in_view_range(bombs_map):
    ret = []
    locations = np.where(bombs_map > 0)
    for r, c in zip(locations[0], locations[1]):
        ret.append({'position': (r, c)})
    return not ret


def flames_in_view_range(board):
    rows = board.shape[0]
    cols = board.shape[1]
    for r in range(0, rows):
        for c in range(0, cols):
            if utility.position_is_flames(board, (r, c)):
                return True
    return False


def enemies_in_view_range(board, enemies):
    rows = board.shape[0]
    cols = board.shape[1]
    for r in range(0, rows):
        for c in range(0, cols):
            if utility.position_is_enemy(board, (r, c), enemies):
                return True
    return False


# Beginning of old dqn.utils
def init_hidden_states(batch_size, hidden_size):
    h = torch.zeros(1, batch_size, hidden_size)
    c = torch.zeros(1, batch_size, hidden_size)

    return h, c


def preprocess_obs(obs):
    board = np.array(obs['board'])
    bomb_blast_strength = np.array(obs['bomb_blast_strength'])
    bomb_life = np.array(obs['bomb_life'])
    bomb_moving_direction = np.array(obs['bomb_moving_direction'])
    flame_life = np.array(obs['flame_life'])

    return np.dstack((board, bomb_blast_strength, bomb_life, bomb_moving_direction, flame_life))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_sample(sample):
    # batch: [[(s, a, r, n_s, d), (s, a, r, n_s, d)], [(s, a, r, n_s, d), (s, a, r, n_s, d)]]
    s_a_n_s_batch, r_d_batch = [], []

    # episode: [(s, a, r, n_s, d), (s, a, r, n_s, d)]
    for episode in sample:
        s_a_n_s_episode, r_d_episode = [], []
        # experience: (s, a, r, n_s, d)
        for experience in episode:
            s_a_n_s_episode.append((experience[0], experience[1], experience[3]))
            r_d_episode.append((experience[2], experience[4]))
        s_a_n_s_batch.append(s_a_n_s_episode)
        r_d_batch.append(r_d_episode)

    return s_a_n_s_batch, r_d_batch


def get_arena_batch(r_d_batch, agent_indices):
    # r is array
    # r_d_batch: [[(r, d), (r, d)], [(r, d), (r, d)]]
    arena_batch = []

    # episode: [(r, d), (r, d)]
    for episode in r_d_batch:
        arena_episode = []
        # experience: (r, d)
        for experience in episode:
            arena_episode.append(get_r_d_experience(experience[0], experience[1], agent_indices))
        arena_batch.append(arena_episode)

    return arena_batch


def get_agent_batch(s_a_n_s_batch, index):
    # s, a, n_s are arrays
    # s_a_n_s_batch: [[(s, a, n_s), (s, a, n_s)], [(s, a, n_s), (s, a, n_s)]]
    agent_batch = []

    # episode: [(s, a, n_s), (s, a, n_s)]
    for episode in s_a_n_s_batch:
        agent_episode = []
        # experience: (s, a, n_s)
        for experience in episode:
            agent_episode.append(get_s_a_n_s_experience(experience[0], experience[1], experience[2], index))
        agent_batch.append(agent_episode)

    return agent_batch


def get_s_a_n_s_experience(state, actions, next_state, index):
    agent_state = state[index]
    agent_action = actions[index]
    agent_next_state = next_state[index]
    experience = (agent_state, agent_action, agent_next_state)
    return experience


def get_r_d_experience(reward, done, agent_indices):
    index_agent1 = agent_indices[0]
    index_agent2 = agent_indices[1]
    assert reward[index_agent1] == reward[index_agent2]
    agent_reward = reward[index_agent1]
    experience = (agent_reward, done)
    return experience
