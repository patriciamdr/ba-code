# Run n models m times
import argparse
import os
from datetime import datetime

from evaluation.constants import GameSetUp
from pommerman.dqn import utils

from pommerman import utility, agents

import pommerman
from pommerman.agents import CommDQNAgentXP


def run(index, _args, _agents, _result_dict):
    record_json_dir = os.path.join(_result_dict, 'json-battle-' + str(index) + datetime.now().isoformat())
    if not os.path.exists(record_json_dir):
        os.makedirs(record_json_dir)

    """Wrapper to help start the game"""
    config = _args.config

    env = pommerman.make(config, _agents)

    print("Starting the Game between {}".format(" and ".join(args.agents.value[:2])))

    state = env.reset()
    reward = None
    done = False

    while not done:
        env.save_json(record_json_dir)
        actions = env.act(state)
        state, reward, done, info = env.step(actions)

    env.save_json(record_json_dir)
    finished_at = datetime.now().isoformat()
    print(_args.config)
    utility.join_json_state(record_json_dir, _args.agents.value, finished_at, _args.config, info)

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
        '--remote',
        default=0,
        type=int)
    parser.add_argument(
        '--num_models',
        default=5,
        type=int
    )
    parser.add_argument(
        '--num_times',
        default=5,
        type=int
    )
    args = parser.parse_args()

    home_directory = os.path.expanduser('~')
    result_path = os.path.join(home_directory, 'dev/playground/json_games')
    team_dir = utils.get_path_from_setting(args.agents.value, result_path)

    # TODO: Adapt agents to agents param

    for model in range(0, args.num_models):
        trial = utils.randomly_select_trained_team(comm=True)
        team_trial_dir = os.path.join(team_dir, trial.split('/')[-1])
        print(team_trial_dir)
        if not os.path.exists(team_trial_dir):
            os.makedirs(team_trial_dir)

        device = utils.get_device()

        for i in range(0, args.num_times):
            team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=trial, random_agent=1, resume_training=0)

            _agents = [
                CommDQNAgentXP(**team_kwargs[0], device=device),
                agents.RandomAgent(),
                CommDQNAgentXP(**team_kwargs[1], device=device),
                agents.RandomAgent()
            ]
            run(i, args, _agents, team_trial_dir)
