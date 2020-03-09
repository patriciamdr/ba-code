import argparse
import os
from ast import literal_eval

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import pommerman

from evaluation import constants
from pommerman.dqn import utils

home_directory = os.path.expanduser('~')
result_path = os.path.join(home_directory, 'dev/playground/battle_csv')


def calculate_success_rate():
    for setting in constants.GameSetUp:
        dict_name = utils.get_path_from_setting(setting.value, result_path)
        csv_path = os.path.join(dict_name, 'results.csv')
        if not os.path.exists(csv_path):
            continue
        print(csv_path)

        data = pd.read_csv(csv_path)
        # Convert reward string to array
        # Convert agents tuple to array
        data["Rewards"] = data["Rewards"].apply(literal_eval)
        data["Agents"] = data["Agents"].apply(literal_eval)
        # First index of Rewards is result of first team, second index of Rewards is result of second team
        data[str(setting.value[0])] = data.Rewards.map(lambda x: x[0] if x[0] == 1 else 0)
        data[str(setting.value[1])] = data.Rewards.map(lambda x: x[1] if x[1] == 1 else 0)
        # Add total of wins
        data.at["Total_wins", str(setting.value[0])] = data[str(setting.value[0])].sum()
        data.at["Total_wins", str(setting.value[1])] = data[str(setting.value[1])].sum()
        # Calculate success rate
        data.at["Success_rate", str(setting.value[0])] = (data.at["Total_wins",
                                                                  str(setting.value[0])] / constants.run_battle_num_times) * 100
        data.at["Success_rate", str(setting.value[1])] = (data.at["Total_wins",
                                                                  str(setting.value[1])] / constants.run_battle_num_times) * 100
        final_path = csv_path.replace('.', '_copy.')
        print(final_path)
        data.to_csv(final_path, index=True, mode='w')


def do_qualitative_analysis():
    battles_with_comm_xp = [setting for setting in constants.GameSetUp if 'xp::comm' in setting.value]
    for setting in battles_with_comm_xp:
        dict_name = utils.get_path_from_setting(setting.value, result_path)
        csv_path = os.path.join(dict_name, 'state_message_pairs.csv')
        if not os.path.exists(csv_path):
            continue
        print(csv_path)

        data = pd.read_csv(csv_path, header=None, names=["Episode", "Scenario", "Message"])
        final_data = pd.DataFrame(columns=["Episode", "Scenario", "Message", "Frequency"])
        for episode in range(0, constants.run_battle_num_times):
            df = data[data.Episode == episode]

            for scenario in constants.Scenario:
                tmp_df = df[df.Scenario == scenario.value]
                frequency = np.arange(pommerman.constants.RADIO_VOCAB_SIZE + 1)
                for _, row in tmp_df.iterrows():
                    frequency[row["Message"]] += 1

                for index, _frequency in enumerate(frequency):
                    final_data.loc[len(final_data)] = [episode, scenario, index, _frequency]

        # Add std
        g = sb.catplot(data=final_data, x='Message', y='Frequency', col='Scenario', kind='bar', ci=None,
                       sharex=False, sharey=False)
        fig_name = 'state_message_dist.svg'
        setting_directory = utils.get_path_from_setting(setting.value, '/home/patricia/GoogleDrive/BA/latex/text/pictures')
        if not os.path.exists(setting_directory):
            os.mkdir(setting_directory)
        fig_path = os.path.join(setting_directory, fig_name)
        # plt.savefig(fig_path, format='svg')
        plt.show()


def analyse_message_action_co_occurrence(prefix, metric_name):
    battles_with_comm_xp = [setting for setting in constants.GameSetUp if 'xp::comm' in setting.value]
    for setting in battles_with_comm_xp:
        dict_name = utils.get_path_from_setting(setting.value, result_path)
        csv_path = os.path.join(dict_name, prefix + '_message_action_pairs.csv')
        if not os.path.exists(csv_path):
            continue
        print(csv_path)

        data = pd.read_csv(csv_path, header=None, names=["Episode", "Message", "Action"])
        data_copy = data.copy()
        for episode in range(0, constants.run_battle_num_times):
            df = data[data.Episode == episode]
            print(df)
            co_occurrence = np.zeros([pommerman.constants.RADIO_VOCAB_SIZE + 1, len(pommerman.constants.Action)])
            for index, row in df.iterrows():
                co_occurrence[row["Message"], row["Action"]] += 1

            p_am = co_occurrence / np.sum(co_occurrence)
            p_a = np.sum(co_occurrence, axis=0) / np.sum(co_occurrence)
            p_m = np.sum(co_occurrence, axis=1) / np.sum(co_occurrence)
            mut_info = 0
            for a in pommerman.constants.Action:
                for m in range(0, pommerman.constants.RADIO_VOCAB_SIZE + 1):
                    if p_am[m, a.value] > 0:
                        mut_info += p_am[m, a.value] * np.log(p_am[m, a.value] / (p_a[a.value] * p_m[m]))

            data_copy.at[metric_name + "-" + str(episode), "Value"] = mut_info

        data_copy.at["Avg_" + metric_name, "Value"] = data_copy.filter(regex=metric_name + "_*", axis=0)["Value"].mean()
        data_copy.at["Std" + metric_name, "Value"] = data_copy.filter(regex=metric_name + "_*", axis=0)["Value"].std()
        final_path = csv_path.replace('.', '_copy.')
        print(final_path)
        data_copy.to_csv(final_path, index=True, mode='w')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calculate_success_rate',
                        default=0,
                        type=int)
    parser.add_argument('--do_qualitative_analysis',
                        default=0,
                        type=int)
    parser.add_argument('--analyse_speaker_consistency',
                        default=0,
                        type=int)
    parser.add_argument('--analyse_ic',
                        default=0,
                        type=int)
    args = parser.parse_args()

    if args.calculate_success_rate:
        calculate_success_rate()
    if args.do_qualitative_analysis:
        do_qualitative_analysis()
    if args.analyse_speaker_consistency:
        analyse_message_action_co_occurrence("sc", "Speaker_consistency")
    if args.analyse_ic:
        analyse_message_action_co_occurrence("ic", "Instantaneous_coordination")
