import argparse
import os

import pandas as pd
from pommerman.dqn import utils

import pommerman
from pommerman import agents
from pommerman.arena_with_comm import CommArenaNOXP
from pommerman.arena import ArenaNOXP


def create_agents(_args):
    # Relevant args for agents
    args_dict = vars(_args)
    agent_kwargs = {key: args_dict[key]
                    for key in ['epsilon', 'tau']}

    device = utils.get_device()
    agent_kwargs['device'] = device

    # Order so that dqn agents will be in one team
    if _args.with_comm:
        agent_list = [
            agents.CommDQNAgentNOXP(**agent_kwargs),
            agents.RandomAgent(),
            agents.CommDQNAgentNOXP(**agent_kwargs),
            agents.RandomAgent()
        ]
    else:
        agent_list = [
            agents.DQNAgentNOXP(**agent_kwargs),
            agents.RandomAgent(),
            agents.DQNAgentNOXP(**agent_kwargs),
            agents.RandomAgent()
        ]

    return agent_list


def create_arena(_args, env, agent_list):
    # Relevant args for arena
    epsilon_decay = (_args.epsilon - _args.final_epsilon) / _args.num_episodes
    args_dict = vars(_args)
    arena_kwargs = {key: args_dict[key]
                    for key in ['tau', 'epsilon', 'discount', 'final_epsilon']}

    device = utils.get_device()
    arena_kwargs['device'] = device
    arena_kwargs['env'] = env
    arena_kwargs['agent_list'] = agent_list
    arena_kwargs['epsilon_decay'] = epsilon_decay

    if _args.with_comm:
        arena = CommArenaNOXP(**arena_kwargs)
    else:
        arena = ArenaNOXP(**arena_kwargs)
    return arena


def run_trial(_args, trial_result_path):
    # Print all possible environments in the Simple-Q-Learning registry
    print(pommerman.REGISTRY)

    # Create a set of agents (exactly four)
    agent_list = create_agents(_args)

    # Make the "Simple-Q-Learning" environment using the agent list
    env = pommerman.make(_args.config, agent_list)

    # Log data
    new_data = pd.DataFrame()

    # Controller for VDN
    arena = create_arena(_args, env, agent_list)

    # Run the episodes just like OpenAI Gym
    episode_rewards = []
    for i_episode in range(_args.num_episodes):
        episode_reward = arena.run_episode()

        df = pd.DataFrame([{'Episode:': i_episode,
                            'Algorithm': 'DQN-NOXP',
                            'Rewards': episode_reward,
                            'Discount': _args.discount,
                            'Initial_Epsilon': _args.epsilon,
                            'Final_Epsilon': _args.final_epsilon,
                            'Tau': _args.tau}])
        new_data = new_data.append(df, ignore_index=True)

        print('Episode {} finished with reward {}'.format(i_episode, episode_reward))
        episode_rewards.append(episode_reward)

    print('Total episode rewards: {}'.format(episode_rewards))
    new_data.to_csv(trial_result_path, index=False, mode='a+')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        # default='PommeTeamCompetition-v0')
                        default='PommeRadioCompetition-v2')
    parser.add_argument('--num_trials',
                        default=2,
                        type=int)
    parser.add_argument('--num_episodes',
                        default=3,
                        type=int)
    parser.add_argument('--discount',
                        default=0.9,
                        type=float)
    parser.add_argument('--epsilon',
                        default=1.0,
                        type=float)
    parser.add_argument('--final_epsilon',
                        default=0.10,
                        type=float)
    parser.add_argument('--tau',
                        default=0.001,
                        type=float)
    parser.add_argument('--with_comm',
                        default=True,
                        type=bool)
    args = parser.parse_args()

    # result_path = '/home/patricia/dev/playground/csv/dqn_noxp'
    result_path = '/home/g/gschossmann/dev/ba-dqn-pommerman/csv/new/dqn_noxp'

    for i in range(args.num_trials):
        file_name = 'trial_' + str(i) + '.csv'
        full_path = os.path.join(result_path, file_name)
        print(full_path)
        run_trial(args, full_path)
