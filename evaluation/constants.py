from enum import Enum

run_battle_num_times = 5
test_epsilon = 0


class GameSetUp(Enum):
    simple_vs_simple = ['Simple-Agent', 'Simple-Agent', 'Simple-Agent', 'Simple-Agent']
    xp_vs_xp = ['No-Comm-Agent', 'No-Comm-Agent', 'No-Comm-Agent', 'No-Comm-Agent']
    comm_xp_vs_comm_xp = ['Comm-Agent', 'Comm-Agent', 'Comm-Agent', 'Comm-Agent']
    random_vs_random = ['Random-Agent', 'Random-Agent', 'Random-Agent', 'Random-Agent']
    comm_xp_vs_xp = ['Comm-Agent', 'No-Comm-Agent', 'Comm-Agent', 'No-Comm-Agent']
    comm_xp_vs_simple = ['Comm-Agent', 'Simple-Agent', 'Comm-Agent', 'Simple-Agent']
    comm_xp_vs_random = ['Comm-Agent', 'Random-Agent', 'Comm-Agent', 'Random-Agent']
    xp_vs_simple = ['No-Comm-Agent', 'Simple-Agent', 'No-Comm-Agent', 'Simple-Agent']
    xp_vs_random = ['No-Comm-Agent', 'Random-Agent', 'No-Comm-Agent', 'Random-Agent']
    random_vs_simple = ['Random-Agent', 'Simple-Agent', 'Random-Agent', 'Simple-Agent']


class Scenario(Enum):
    bomb_in_view_range = "bombInViewRange"
    flames_in_view_range = "flamesInViewRange"
    enemies_in_view_range = "enemiesInViewRange"
