import argparse
import os
from ast import literal_eval
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


def evaluate_mut_info(_dir, path, _figure_name):
    full_path = os.path.join(path, _dir)
    files = [os.path.join(root, file) for root, _, files in os.walk(full_path) for file in files if file.endswith('.csv')]

    overall_data = []
    for file in files:
        data = pd.read_csv(file)

        # Smooth trial over every 100 episodes
        kernel = np.ones(100)/100
        data["Smoothed_SC"] = np.convolve(data["SC"], kernel, mode='same')
        data["Smoothed_IC"] = np.convolve(data["IC"], kernel, mode='same')

        data = data[["Episode", "Algorithm", "Smoothed_SC", "Smoothed_IC"]]
        overall_data.append(data)
    df = pd.concat(overall_data, axis=0)
    sb.set_style('whitegrid', {'grid.linestyle': '--'})
    sb.set_style('ticks')
    sb.set_context('paper')

    training_directory = '/home/patricia/GoogleDrive/BA/latex/text/pictures/training'
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)

    g = sb.lineplot(x='Episode', y='Smoothed_SC', data=df, ci=95)
    _figure_svg = _figure_name + '_sc.svg'
    _figure_png = _figure_name + '_sc.png'
    fig_path_svg = os.path.join(training_directory, _figure_svg)
    fig_path_png = os.path.join(training_directory, _figure_png)
    plt.xlabel('Episode')
    plt.ylabel('Speaker Consistency')
    plt.show()
    # plt.savefig(fig_path_svg, format='svg')
    # plt.savefig(fig_path_png, format='png')

    g = sb.lineplot(x='Episode', y='Smoothed_IC', data=df, ci=95)
    _figure_svg = _figure_name + '_ic.svg'
    _figure_png = _figure_name + '_ic.png'
    fig_path_svg = os.path.join(training_directory, _figure_svg)
    fig_path_png = os.path.join(training_directory, _figure_png)
    plt.xlabel('Episode')
    plt.ylabel('Instantaneous Coordination')
    plt.show()
    # plt.savefig(fig_path_svg, format='svg')
    # plt.savefig(fig_path_png, format='png')


def evaluate_training(_dir, path, _figure_name):
    combined_df = pd.DataFrame()
    for d in _dir:
        full_path = os.path.join(path, d)
        files = [os.path.join(root, file) for root, _, files in os.walk(full_path) for file in files if file.endswith('.csv')]

        overall_data = []
        for file in files:
            data = pd.read_csv(file, dtype={"Episode": object, "Algorithm": object, "Opponent": object,
                                            "Rewards": object, "SC": object, "IC": object,
                                            "Discount": object, "Initial_Epsilon": object,
                                            "Final_Epsilon": object, "Tau": object, "Time_step": object,
                                            "Batch_size": object})

            # Convert reward string to array
            data["Rewards"] = data["Rewards"].apply(literal_eval)
            # Save result: -1 for loss, 0 for tie, 1 for win
            # First index of Rewards is reward of DQN-Team
            # data["Result"] = data.Rewards.map(lambda x: x[0] if sum(x) == 0 else 0)
            data["Reward"] = data.Rewards.map(lambda x: x[0])
            data["Win"] = data.Rewards.map(lambda x: x[0] if x[0] == 1 else 0)

            # Smooth trial over every 100 episodes
            kernel = np.ones(100)/100
            data["Smoothed_Reward"] = np.convolve(data["Reward"], kernel, mode='same')

            data = data[["Episode", "Algorithm", "Smoothed_Reward", "Win"]]
            data["Episode"] = data["Episode"].astype(str).astype(int)
            overall_data.append(data)
        df = pd.concat(overall_data, axis=0)
        combined_df = combined_df.append(df, sort=False)
    sb.set_style('whitegrid', {'grid.linestyle': '--'})
    sb.set_style('ticks')
    sb.set_context('paper')

    training_directory = '/home/patricia/GoogleDrive/BA/latex/text/pictures/training'
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)

    g = sb.lineplot(x='Episode', y='Smoothed_Reward', hue='Algorithm', data=combined_df, ci=95)
    g.legend().texts[0].set_text('Modell')
    _figure_svg = _figure_name + '_result.svg'
    _figure_png = _figure_name + '_result.png'
    fig_path_svg = os.path.join(training_directory, _figure_svg)
    fig_path_png = os.path.join(training_directory, _figure_png)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    # plt.savefig(fig_path_svg, format='svg')
    # plt.savefig(fig_path_png, format='png')

    g = sb.lineplot(x='Episode', y='Win', hue='Algorithm', estimator=lambda x: mean(x) * 100, data=combined_df, ci=None)
    g.legend().texts[0].set_text('Modell')
    _figure_svg = _figure_name + '_win.svg'
    _figure_png = _figure_name + '_win.png'
    fig_path_svg = os.path.join(training_directory, _figure_svg)
    fig_path_png = os.path.join(training_directory, _figure_png)
    plt.ylabel("Gewinnrate")
    plt.show()
    # plt.savefig(fig_path_svg, format='svg')
    # plt.savefig(fig_path_png, format='png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sc_and_ic',
                        default=1,
                        type=int)
    args = parser.parse_args()

    home_directory = os.path.expanduser('~')
    result_path = os.path.join(home_directory, 'dev/playground/csv')

    figure_name = 'learning_curve'
    mut_info_figure_name = 'comm_mut_info'

    comm_dir = 'comm_xp_results'
    no_comm_dir = 'xp_results'

    evaluate_training([comm_dir, no_comm_dir], result_path, figure_name)
    if args.sc_and_ic:
        evaluate_mut_info(comm_dir, result_path, mut_info_figure_name)
