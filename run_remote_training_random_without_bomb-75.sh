#!/usr/bin/env bash
source venv/bin/activate
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmNjNjM4MDYtZWU4MS00Nzg2LWIzZWEtMWQzNDM5OGZlNWYyIn0=
# export TMPDIR=~/tmp
# pip install .
# Params are trial_number, num_episodes, opponent, resume_training, with_comm, reward_shaping
for index in {51..58}; do
  ./exec_training.sh "$index" 10000 'RandomAgentWithoutBomb' 0 1 1
  ./exec_training.sh "$index" 10000 'RandomAgentWithoutBomb' 0 0 1
done
