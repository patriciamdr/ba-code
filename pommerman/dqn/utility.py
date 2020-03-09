from pommerman.dqn import utils
from pommerman import agents
from pommerman.agents import CommDQNAgentXP, DQNAgentXP, RandomAgent, RandomAgentWithoutBomb


def simple_vs_simple_func():
    print("Simple vs. simple")
    return [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]


def comm_xp_vs_comm_xp_func():
    print("CommXP vs. CommXP")
    # TODO: Change which model to load if played against other dqn agent
    fst_team_dir = utils.randomly_select_trained_team(True)
    fst_team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=fst_team_dir, random_agent=1, resume_training=0)
    snd_team_dir = utils.randomly_select_trained_team(True)
    snd_team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=snd_team_dir, random_agent=1, resume_training=0)
    return [
        CommDQNAgentXP(**fst_team_kwargs[0]),
        CommDQNAgentXP(**snd_team_kwargs[0]),
        CommDQNAgentXP(**fst_team_kwargs[1]),
        CommDQNAgentXP(**snd_team_kwargs[1]),
    ]


def xp_vs_xp_func():
    print("XP vs. XP")
    # TODO: Change which model to load if played against other dqn agent
    fst_team_dir = utils.randomly_select_trained_team(False)
    fst_team_kwargs, _ = utils.load_trained_team(comm=0, team_dir=fst_team_dir, random_agent=1, resume_training=0)
    snd_team_dir = utils.randomly_select_trained_team(False)
    snd_team_kwargs, _ = utils.load_trained_team(comm=0, team_dir=snd_team_dir, random_agent=1, resume_training=0)
    return [
        DQNAgentXP(**fst_team_kwargs[0]),
        DQNAgentXP(**snd_team_kwargs[0]),
        DQNAgentXP(**fst_team_kwargs[1]),
        DQNAgentXP(**snd_team_kwargs[1]),
    ]


def random_vs_random_func():
    return [
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
    ]


def comm_xp_vs_xp_func():
    print("CommXP vs. XP")
    # TODO: Change which model to load if played against other dqn agent
    fst_team_dir = utils.randomly_select_trained_team(True)
    fst_team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=fst_team_dir, random_agent=1, resume_training=0)
    snd_team_dir = utils.randomly_select_trained_team(False)
    snd_team_kwargs, _ = utils.load_trained_team(comm=0, team_dir=snd_team_dir, random_agent=1, resume_training=0)
    return [
        CommDQNAgentXP(**fst_team_kwargs[0]),
        DQNAgentXP(**snd_team_kwargs[0]),
        CommDQNAgentXP(**fst_team_kwargs[1]),
        DQNAgentXP(**snd_team_kwargs[1]),
    ]


def comm_xp_vs_simple_func():
    print("CommXP vs. simple")
    team_dir = utils.randomly_select_trained_team(True)
    team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=team_dir, random_agent=0, resume_training=0)
    return [
        CommDQNAgentXP(**team_kwargs[0]),
        agents.SimpleAgent(),
        CommDQNAgentXP(**team_kwargs[1]),
        agents.SimpleAgent()
    ]


def comm_xp_vs_random_func():
    print("CommXP vs. random")
    team_dir = utils.randomly_select_trained_team(True)
    team_kwargs, _ = utils.load_trained_team(comm=1, team_dir=team_dir, random_agent=1, resume_training=0)
    return [
        CommDQNAgentXP(**team_kwargs[0]),
        agents.RandomAgent(),
        CommDQNAgentXP(**team_kwargs[1]),
        agents.RandomAgent()
    ]


def xp_vs_simple_func():
    print("XP vs. simple")
    team_dir = utils.randomly_select_trained_team(False)
    team_kwargs, _ = utils.load_trained_team(comm=0, team_dir=team_dir, random_agent=0, resume_training=0)
    return [
        DQNAgentXP(**team_kwargs[1]),
        agents.SimpleAgent(),
        DQNAgentXP(**team_kwargs[1]),
        agents.SimpleAgent()
    ]


def xp_vs_random_func():
    print("XP vs. random")
    team_dir = utils.randomly_select_trained_team(comm=False)
    team_kwargs, _ = utils.load_trained_team(comm=0, team_dir=team_dir, random_agent=1, resume_training=0)
    return [
        DQNAgentXP(**team_kwargs[0]),
        agents.RandomAgent(),
        DQNAgentXP(**team_kwargs[1]),
        agents.RandomAgent()
    ]


def random_vs_simple_func(_args):
    print("Random vs. simple")
    return [
        agents.RandomAgent(),
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.SimpleAgent(),
    ]


def create_agents(index_agent_dict, opponent):
    if opponent == 'RandomAgent':
        opponents = [RandomAgent(), RandomAgent()]
    elif opponent == 'RandomAgentWithoutBomb':
        opponents = [RandomAgentWithoutBomb(), RandomAgentWithoutBomb()]
    else:
        device = utils.get_device()

        opponents_dir = utils.randomly_select_trained_team(comm=True)
        opponents_kwargs, _ = utils.load_trained_team(comm=1, team_dir=opponents_dir, random_agent=1, resume_training=0)
        opponents = [
            CommDQNAgentXP(**opponents_kwargs[0], device=device),
            CommDQNAgentXP(**opponents_kwargs[1], device=device),
        ]

    agent_list = [
        index_agent_dict.get(0),
        opponents[0],
        index_agent_dict.get(2),
        opponents[1]
    ]
    return agent_list


def get_shaped_reward(reward):
    if reward == [-1, -1, -1, -1]:
        return [0, 0, 0, 0]
    else:
        return reward
