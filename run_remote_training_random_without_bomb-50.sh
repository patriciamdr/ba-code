#!/usr/bin/env bash
source venv/bin/activate
# pip install .
# Params are trial_number, num_episodes, opponent, resume_training, with_comm, reward_shaping
for index in {25..50}; do
  sbatch --partition=All --cpus-per-task=4 ./exec_training.sh "$index" 10000 'RandomAgentWithoutBomb' 0 1 1
  sbatch --partition=All --cpus-per-task=4 ./exec_training.sh "$index" 10000 'RandomAgentWithoutBomb' 0 0 1
done
