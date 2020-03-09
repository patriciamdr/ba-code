#!/usr/bin/env bash
python examples/dqn_test_run_xp.py --trial_number="$1" --num_episodes="$2" --opponent="$3" --resume_training="$4" --with_comm="$5" --reward_shaping="$6" --remote=1
