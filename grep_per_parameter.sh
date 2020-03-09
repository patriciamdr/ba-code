#!/usr/bin/env bash
# 4999,DQN-NOXP,"[-1, 1, -1, 1]",1.0,1.0,0.1,0.08
discounts=(0.99 0.999 1)
taus=(0.001 0.05 0.08 0.7)
time_steps=(2 4 8)

num_trial=0
for trial in csv/new/dqn_noxp/*.csv
do
  num_trial=$((num_trial+1))
  count=$(grep -c -E ".+,DQN-NOXP,.+,0\.999,1\.0,0\.1,0\.7" "$trial")
  if [ 5000 -ne "$count" ]; then
    echo "$trial|$count"
  fi
done
echo "$num_trial"
