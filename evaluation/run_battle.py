# Run n battles between agents, models are randomly chosen for each episode
import argparse
import os

import neptune

import numpy as np
import pandas as pd

from evaluation.constants import GameSetUp, run_battle_num_times, Scenario, test_epsilon
from pommerman.dqn import utils, utility

from pommerman import agents, constants

import pommerman


def create_agents(_args, agent_strings):
    switcher = {
        GameSetUp.simple_vs_simple: utility.simple_vs_simple_func,
        GameSetUp.xp_vs_xp: utility.xp_vs_xp_func,
        GameSetUp.comm_xp_vs_comm_xp: utility.comm_xp_vs_comm_xp_func,
        GameSetUp.random_vs_random: utility.random_vs_random_func,
        GameSetUp.comm_xp_vs_xp: utility.comm_xp_vs_xp_func,
        GameSetUp.comm_xp_vs_simple: utility.comm_xp_vs_simple_func,
        GameSetUp.comm_xp_vs_random: utility.comm_xp_vs_random_func,
        GameSetUp.xp_vs_random: utility.xp_vs_random_func,
        GameSetUp.xp_vs_simple: utility.xp_vs_simple_func,
        GameSetUp.random_vs_simple: utility.random_vs_simple_func,
    }
    func = switcher.get(agent_strings, "Invalid agents argument")
    return func()


def save_scenario_message_pair(episode, scenario, message, path):
    message_state_data = pd.DataFrame()
    # TODO: Ask Thomy how to evaluate:
    # Message as list or each word as its own?
    for m in message:
        df = pd.DataFrame([{'Episode': episode,
                            'Scenario': scenario,
                            'Message': m}])
        message_state_data = message_state_data.append(df, ignore_index=True)
        message_state_data.to_csv(path, index=False, mode='a+', header=False)


def do_qualitative_analysis(episode, index_comm_agent_dict, save_messages_given_states, state, actions, path):
    if save_messages_given_states:
        for index, _ in index_comm_agent_dict.items():
            obs = state[index]
            board = np.array(obs['board'])
            bombs_map = np.array(obs['bomb_blast_strength'])
            enemies = [constants.Item(e) for e in obs['enemies']]
            action = actions[index]

            # TODO: Think of better scenarios
            if type(action) in [tuple, list]:
                if utils.bombs_in_view_range(bombs_map):
                    save_scenario_message_pair(episode, Scenario.bomb_in_view_range.value, action[1:3], path)
                if utils.flames_in_view_range(board):
                    save_scenario_message_pair(episode, Scenario.flames_in_view_range.value, action[1:3], path)
                if utils.enemies_in_view_range(board, enemies):
                    save_scenario_message_pair(episode, Scenario.enemies_in_view_range.value, action[1:3], path)


def save_message_action_pair(episode, action, message, path):
    message_action_data = pd.DataFrame()
    # TODO: Ask Thomy how to evaluate:
    # Message as list or each word as its own?
    for m in message:
        df = pd.DataFrame([{'Episode': episode,
                            'Message': m,
                            'Action': action}])
        message_action_data = message_action_data.append(df, ignore_index=True)
        message_action_data.to_csv(path, index=False, mode='a+', header=False)


def analyse_speaker_consistency(episode, index_comm_agent_dict, save_speaker_consistency, actions, path):
    if save_speaker_consistency:
        for index, _ in index_comm_agent_dict.items():
            action = actions[index]

            if type(action) in [tuple, list]:
                save_message_action_pair(episode, action[0], action[1:3], path)


def analyse_ic(episode, index_comm_agent_dict, save_instantaneous_coordination, actions, path):
    if save_instantaneous_coordination:
        indices = list(index_comm_agent_dict.keys())
        action_1 = actions[indices[0]]
        action_2 = actions[indices[1]]

        if (type(action_1) in [tuple, list]) and (type(action_2) in [tuple, list]):
            save_message_action_pair(episode, action_1[0], action_2[1:3], path)
            save_message_action_pair(episode, action_2[0], action_1[1:3], path)


def run(index, _args, _result_dict, _write_header):
    """Wrapper to help start the game"""
    config = _args.config

    _agents = create_agents(_args, args.agents)
    index_comm_agent_dict = {index: agent
                             for index, agent in enumerate(_agents) if isinstance(agent, agents.CommDQNAgent)}

    env = pommerman.make(config, _agents)

    csv_path = os.path.join(_result_dict, 'results.csv')
    state_message_pairs_path = os.path.join(_result_dict, 'state_message_pairs.csv')
    sc_message_action_pairs_path = os.path.join(_result_dict, 'sc_message_action_pairs.csv')
    ic_message_action_pairs_path = os.path.join(_result_dict, 'ic_message_action_pairs.csv')

    print("Starting the Game between {}".format(" and ".join(args.agents.value[:2])))

    state = env.reset()
    reward = None
    done = False

    while not done:
        if _args.render:
            env.render()
        actions = env.act(state)
        do_qualitative_analysis(index, index_comm_agent_dict, _args.save_messages_given_states,
                                state, actions, state_message_pairs_path)
        analyse_speaker_consistency(index, index_comm_agent_dict, _args.save_speaker_consistency, actions,
                                    sc_message_action_pairs_path)
        analyse_ic(index, index_comm_agent_dict, _args.save_instantaneous_coordination, actions,
                   ic_message_action_pairs_path)
        state, reward, done, info = env.step(actions)

    neptune.send_text('Rewards', index, str(reward))

    new_data = pd.DataFrame()
    df = pd.DataFrame([{'Episode': index,
                        'Agents': args.agents.value,
                        'Rewards': reward}])
    new_data = new_data.append(df, ignore_index=True)
    new_data.to_csv(csv_path, index=False, mode='a+', header=_write_header)

    print('Episode {} finished with reward {}'.format(index, reward))
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground Flags.')
    parser.add_argument(
        '--config',
        default='PommeRadioCompetition-v2',
        help='Configuration to execute. See env_ids in '
             'configs.py for options.')
    parser.add_argument(
        '--agents',
        # default=GameSetUp.xp_vs_xp,
        # default=GameSetUp.comm_xp_vs_comm_xp,
        # default=GameSetUp.simple_vs_simple,
        # default=GameSetUp.random_vs_random,
        # default=GameSetUp.comm_xp_vs_xp,
        # default=GameSetUp.comm_xp_vs_simple,
        default=GameSetUp.comm_xp_vs_random,
        # default=GameSetUp.xp_vs_simple,
        # default=GameSetUp.xp_vs_random,
        # default=GameSetUp.random_vs_simple,
        help='Comma delineated list of agent types and docker '
             'locations to run the agents.')
    parser.add_argument(
        '--save_messages_given_states',
        default=0,
        type=int
    )
    parser.add_argument(
        '--save_speaker_consistency',
        default=0,
        type=int
    )
    parser.add_argument(
        '--save_instantaneous_coordination',
        default=0,
        type=int
    )
    parser.add_argument(
        '--remote',
        default=0,
        type=int)
    parser.add_argument(
        '--render',
        default=0,
        type=int
    )
    args = parser.parse_args()

    home_directory = os.path.expanduser('~')
    result_path = os.path.join(home_directory, 'dev/playground/battle_csv')

    result_dict = utils.get_path_from_setting(args.agents.value, result_path)
    if not os.path.exists(result_dict):
        os.makedirs(result_dict)

    PARAMS = {
        'run_battle_num_times': run_battle_num_times,
        'agents': args.agents.value,
        'epsilon': test_epsilon,
        'save_messages_given_states': args.save_messages_given_states,
        'save_speaker_consistency': args.save_speaker_consistency,
        'save_instantaneous_coordination': args.save_instantaneous_coordination
    }

    neptune.init('pgschossmann/ba-dqn-pommerman',
                 api_token=os.environ.get('NEPTUNE_API_TOKEN'))
    neptune.create_experiment(name='run_battle-test', params=PARAMS, description=str(args.agents.value))
    if args.save_messages_given_states:
        neptune.append_tag('save_messages_given_states')
    if args.save_speaker_consistency:
        neptune.append_tag('save_speaker_consistency')
    if args.save_instantaneous_coordination:
        neptune.append_tag('save_instantaneous_coordination')
    if args.remote:
        neptune.append_tag('remote')

    write_header = True
    for i in range(0, run_battle_num_times):
        run(i, args, result_dict, write_header)
        write_header = False

    neptune.stop()
