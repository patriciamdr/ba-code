import argparse
import time
import calendar

import neptune
import os

import pandas as pd
import torch

from pommerman.agents import CommDQNAgentXP, DQNAgentXP
from pommerman.dqn.nets.recurrent_cnn import CommDRCNN, PreprocessingDRCNN

from pommerman import agents, constants
from pommerman.arena_with_comm import CommArenaXP
from pommerman.arena import ArenaXP

from pommerman.dqn import utils


def create_arena(_args, _dqn_agents, _resumed_optimizers=None):
    assert _dqn_agents is not None
    # Relevant args for arena
    epsilon_decay = (PARAMS['epsilon_start'] - PARAMS['epsilon_end']) / _args.num_episodes
    args_dict = vars(_args)
    arena_kwargs = {key: args_dict[key]
                    for key in ['batch_size', 'tau', 'discount', 'time_step']}

    device = utils.get_device()
    arena_kwargs['device'] = device
    arena_kwargs['dqn_agents'] = _dqn_agents
    arena_kwargs['epsilon_decay'] = epsilon_decay
    arena_kwargs['epsilon'] = PARAMS['epsilon_start']
    arena_kwargs['final_epsilon'] = PARAMS['epsilon_end']
    arena_kwargs['reward_shaping'] = PARAMS['reward_shaping']
    if _resumed_optimizers is not None:
        arena_kwargs['resumed_optimizers'] = _resumed_optimizers

    if _args.with_comm:
        arena = CommArenaXP(**arena_kwargs)
    else:
        arena = ArenaXP(**arena_kwargs)
    return arena


def create_dqn_agents(_args, _full_trained_model_path=None):
    # Relevant args for agents
    args_dict = vars(_args)
    agent_kwargs = {key: args_dict[key]
                    for key in ['tau', 'time_step', 'batch_size', 'epsilon']}

    device = utils.get_device()
    agent_kwargs['device'] = device

    if _args.resume_training:
        # Resume training
        assert _full_trained_model_path is not None
        _resumed_agents_kwargs, _resumed_optimizers_dicts = utils.load_trained_team(comm=_args.with_comm,
                                                                                    team_dir=_full_trained_model_path,
                                                                                    random_agent=1,
                                                                                    resume_training=1)

        if _args.with_comm:
            _resumed_agents = [
                CommDQNAgentXP(**_resumed_agents_kwargs[0], **agent_kwargs),
                CommDQNAgentXP(**_resumed_agents_kwargs[1], **agent_kwargs),
            ]

            params = []
            comm_params = []
            # Add agent model params
            for _agent in _resumed_agents:
                agent_params, agent_comm_params = _agent.get_model_params()
                params += agent_params
                comm_params += agent_comm_params

            adam = torch.optim.Adam(params)
            adam.load_state_dict(_resumed_optimizers_dicts[0])

            comm_adam = torch.optim.Adam(comm_params)
            comm_adam.load_state_dict(_resumed_optimizers_dicts[1])

            _resumed_optimizers = [
                adam,
                comm_adam
            ]
        else:
            _resumed_agents = [
                DQNAgentXP(**_resumed_agents_kwargs[0], **agent_kwargs),
                DQNAgentXP(**_resumed_agents_kwargs[1], **agent_kwargs),
            ]

            params = []
            # Add agent model params
            for _agent in _resumed_agents:
                params += _agent.get_model_params()

            adam = torch.optim.Adam(params)
            adam.load_state_dict(_resumed_optimizers_dicts[0])

            _resumed_optimizers = [
                adam
            ]
        return _resumed_agents, _resumed_optimizers
    else:
        if _args.with_comm:
            action_model = CommDRCNN(constants.BOARD_SIZE, len(constants.Action))
            agent_kwargs['action_model'] = action_model
            comm_model = PreprocessingDRCNN(constants.BOARD_SIZE, constants.RADIO_VOCAB_SIZE + 1)
            agent_kwargs['comm_model'] = comm_model
            agent_1 = agents.CommDQNAgentXP(**agent_kwargs)

            action_model = CommDRCNN(constants.BOARD_SIZE, len(constants.Action))
            agent_kwargs['action_model'] = action_model
            comm_model = PreprocessingDRCNN(constants.BOARD_SIZE, constants.RADIO_VOCAB_SIZE + 1)
            agent_kwargs['comm_model'] = comm_model
            agent_2 = agents.CommDQNAgentXP(**agent_kwargs)

            _dqn_agents = [
                agent_1,
                agent_2,
            ]
        else:
            action_model = PreprocessingDRCNN(constants.BOARD_SIZE, len(constants.Action))
            agent_kwargs['action_model'] = action_model
            agent_1 = agents.DQNAgentXP(**agent_kwargs)

            action_model = PreprocessingDRCNN(constants.BOARD_SIZE, len(constants.Action))
            agent_kwargs['action_model'] = action_model
            agent_2 = agents.DQNAgentXP(**agent_kwargs)

            _dqn_agents = [
                agent_1,
                agent_2,
            ]

        return _dqn_agents, None


def run_trial(_args, trial_result_path, _full_trained_model_path=None):
    print('Trial result path: {}'.format(trial_result_path))

    _dqn_agents, _resumed_optimizers = create_dqn_agents(_args, _full_trained_model_path)

    # Controller for VDN
    arena = create_arena(_args, _dqn_agents, _resumed_optimizers)

    # Run the episodes just like OpenAI Gym
    episodes_rewards = []

    write_header = True
    _optimizer_state_dict = None
    _comm_optimizer_state_dict = None
    for i_episode in range(PARAMS['episode_start'], PARAMS['episode_end']):
        episode_reward, sc, ic, _optimizer_state_dict, _comm_optimizer_state_dict = arena.run_episode(
            _args.opponent)

        neptune.send_text('Rewards', i_episode, str(episode_reward))

        # Save episode info - reward, ic, sc
        new_data = pd.DataFrame()
        df = pd.DataFrame([{'Episode': i_episode,
                            'Algorithm': PARAMS['algorithm'],
                            'Opponent': PARAMS['opponent'],
                            'Rewards': episode_reward,
                            'SC': sc,
                            'IC': ic,
                            'Discount': PARAMS['discount'],
                            'Initial_Epsilon': PARAMS['epsilon_start'],
                            'Final_Epsilon': PARAMS['epsilon_end'],
                            'Tau': PARAMS['tau'],
                            'Time_step': PARAMS['time_step'],
                            'Batch_size': PARAMS['batch_size']}])
        new_data = new_data.append(df, ignore_index=True)
        new_data.to_csv(trial_result_path, index=False, mode='a+', header=write_header)
        write_header = False

        print('Episode {} finished with reward {}'.format(i_episode, episode_reward))
        episodes_rewards.append(episode_reward)

    print('Total episodes rewards: {}'.format(episodes_rewards))
    return _dqn_agents, episodes_rewards, _optimizer_state_dict, _comm_optimizer_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes',
                        default=10000,
                        type=int)
    parser.add_argument('--discount',
                        default=0.99,
                        type=float)
    parser.add_argument('--tau',
                        default=0.001,
                        type=float)
    parser.add_argument('--batch_size',
                        default=32,
                        type=int)
    parser.add_argument('--time_step',
                        default=8,
                        type=int)
    parser.add_argument('--with_comm',
                        default=1,
                        type=int)
    parser.add_argument('--opponent')
    # default='RandomAgentWithoutBomb')
    # default='Self')
    # default='RandomAgent')
    parser.add_argument('--resume_training',
                        default=0,
                        type=int)
    parser.add_argument('--reward_shaping',
                        default=0,
                        type=int)
    parser.add_argument('--trial_number',
                        default=None,
                        type=int)
    parser.add_argument('--remote',
                        default=0,
                        type=int)
    parser.add_argument('--epsilon',
                        default=0.1,
                        type=int)
    args = parser.parse_args()

    home_directory = os.path.expanduser('~')
    result_path = os.path.join(home_directory, 'dev/playground/csv')
    trained_model_path = os.path.join(home_directory, 'dev/playground/models')

    if args.with_comm:
        subdirectory = 'comm_xp_results'
        algorithm = 'Comm-DQN'
    else:
        subdirectory = 'xp_results'
        algorithm = 'No-Comm-DQN'

    result_path = os.path.join(result_path, subdirectory)
    trained_model_path = os.path.join(trained_model_path, subdirectory)

    opponent = args.opponent
    if args.resume_training:
        episode_start = 10000
    else:
        episode_start = 0

    PARAMS = {
        'trial_number': args.trial_number,
        'algorithm': algorithm,
        'opponent': opponent,
        'reward_shaping': args.reward_shaping,
        'episode_start': episode_start,
        'episode_end': episode_start + args.num_episodes,
        'discount': args.discount,
        'epsilon_start': args.epsilon,
        'epsilon_end': args.epsilon,
        'tau': args.tau,
        'batch_size': args.batch_size,
        'time_step': args.time_step,
    }

    ts = calendar.timegm(time.gmtime())
    base = str(args.trial_number)

    full_trial_path = os.path.join(result_path, 'trial_' + base)
    full_trained_model_path = os.path.join(trained_model_path, 'trial_' + base)

    if not os.path.exists(full_trial_path):
        os.makedirs(full_trial_path)

    if not os.path.exists(full_trained_model_path):
        os.makedirs(full_trained_model_path)

    neptune.init('pgschossmann/ba-dqn-pommerman',
                 api_token=os.environ.get('NEPTUNE_API_TOKEN'))
    neptune.create_experiment(name='dqn_test_run_xp', params=PARAMS,
                              description=os.path.join(subdirectory, 'trial_' + base))
    neptune.append_tag(algorithm)
    neptune.append_tag('with_metrics_calc')
    neptune.append_tag(args.opponent)
    if args.remote:
        neptune.append_tag('remote')
    if args.resume_training:
        neptune.append_tag('resume_training')

    # epoch-opponent-timestamp
    prefix = str(PARAMS['episode_end']) + '-' + opponent
    file_name = prefix + '-' + str(ts) + '.csv'
    full_path = os.path.join(full_trial_path, file_name)

    dqn_agents, episode_rewards, optimizer_state_dict, comm_optimizer_state_dict = run_trial(args, full_path,
                                                                                             full_trained_model_path)
    for i, agent in enumerate(dqn_agents):
        if isinstance(agent, agents.CommDQNAgent):
            torch.save({
                'action_model_' + str(i): agent.action_model.state_dict(),
                'comm_model_' + str(i): agent.comm_model.state_dict(),
            }, os.path.join(full_trained_model_path, prefix + '-agent_' + str(i) + '.tar'))
        else:
            torch.save({
                'action_model_' + str(i): agent.action_model.state_dict(),
            }, os.path.join(full_trained_model_path, prefix + '-agent_' + str(i) + '.tar'))

    torch.save({
        'comm_optimizer': comm_optimizer_state_dict,
        'optimizer': optimizer_state_dict,
    }, os.path.join(full_trained_model_path, prefix + '-optimizers' + '.tar'))
    neptune.stop()
