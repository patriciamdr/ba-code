#!/usr/bin/env bash

source ./venv/bin/activate
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmNjNjM4MDYtZWU4MS00Nzg2LWIzZWEtMWQzNDM5OGZlNWYyIn0=
# export TMPDIR=~/tmp
# pip install .

for index in {51..75}; do
	python examples/dqn_test_run_xp.py --trial_number="$index" --num_episodes=10000 --opponent=RandomAgentWithoutBomb --resume_training=0 --with_comm=0 --reward_shaping=1
	python examples/dqn_test_run_xp.py --trial_number="$index" --num_episodes=10000 --opponent=RandomAgentWithoutBomb --resume_training=1 --with_comm=0 --reward_shaping=1
done
