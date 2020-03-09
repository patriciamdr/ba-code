#!/usr/bin/env bash
source venv/bin/activate
pip install .
# Params are trial_number, num_episodes, opponent, resume_training, with_comm, reward_shaping
for index in {0..50}; do
  sbatch --partition=Gobi ./exec_training.sh "$index" 10000 'RandomAgent' 0 1 1
  sbatch --partition=Gobi ./exec_training.sh "$index" 10000 'RandomAgent' 0 0 1
done
