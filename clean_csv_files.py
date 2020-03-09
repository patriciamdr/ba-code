import os

import pandas as pd


def clean_csv_files():
    comm_xp_dir = '/home/patricia/dev/playground/csv/old/random_agent/comm_xp_results'
    xp_dir = '/home/patricia/dev/playground/csv/old/random_agent/xp_results'

    _dir = [comm_xp_dir, xp_dir]
    for d in _dir:
        num_files = len([name for name in os.listdir(d) if name != '.blank'])
        for file in range(0, num_files):
            file_name = 'trial_' + str(file) + '.csv'
            full_path = os.path.join(d, file_name)
            data = pd.read_csv(full_path)

            last_trial_discount99 = data[data.Discount == str(0.99)]
            last_trial_discount99 = last_trial_discount99.reset_index()
            idx = last_trial_discount99[last_trial_discount99.Episode == str(0)].index.max()
            last_trial_discount99 = last_trial_discount99.iloc[idx:]

            last_trial_discount999 = data[data.Discount == str(0.999)]
            last_trial_discount999 = last_trial_discount999.reset_index()
            idx = last_trial_discount999[last_trial_discount999.Episode == str(0)].index.max()
            last_trial_discount999 = last_trial_discount999.iloc[idx:]

            final_data = last_trial_discount99.append(last_trial_discount999)
            final_data.to_csv(full_path, index=False, mode='w')


if __name__ == '__main__':
    clean_csv_files()
