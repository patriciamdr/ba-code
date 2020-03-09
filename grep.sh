#!/usr/bin/env bash
# 4999,DQN-NOXP,"[-1, 1, -1, 1]",1.0,1.0,0.1,0.08
discounts=(0.99 0.999 1)
taus=(0.001 0.05 0.08 0.7)

for trial in csv/new/dqn_noxp/*.csv
do
  for d in "${discounts[@]}"
  do
    for t in "${taus[@]}"
    do
      count=$(grep -c -E ".+,DQN-NOXP,.+,$d,1\.0,0\.1,$t" "$trial")
      if [ 5000 -ne "$count" ]; then
        echo "$trial|$d|$t|$count"
      fi
    done
  done
done

# 0,DQN_XP,"[1, -1, 1, -1]",0.99,1.0,0.1,0.001,2,32
time_steps=(2 4 8)

for trial in csv/new/dqn_xp/*.csv
do
  for d in "${discounts[@]}"
  do
    for t in "${taus[@]}"
    do
      for ts in "${time_steps[@]}"
      do
        count=$(grep -c -E ".+,DQN_XP,.+,$d,1\.0,0\.1,$t,$ts,32" "$trial")
        if [ 5000 -ne "$count" ]; then
          echo "$trial|$d|$t|$ts|$count"
        fi
      done
    done
  done
done
