#!/usr/bin/env bash
set -euo pipefail

cwd=$(pwd)
CKPT_ROOT="$cwd/checkpoints"
export PYTHONPATH="$cwd:${PYTHONPATH:-}"

PLANNER="planTF"
SPLIT="${1:-test14-hard}"

CHALLENGES=(
  "closed_loop_nonreactive_agents"
  "closed_loop_reactive_agents"
  "open_loop_boxes"
)

WORKERS_PER_JOB="${2:-8}"

mkdir -p logs

for challenge in "${CHALLENGES[@]}"; do
    echo "[START] $challenge"

    CUDA_VISIBLE_DEVICES=0 python run_simulation.py \
        +simulation="$challenge" \
        planner="$PLANNER" \
        scenario_builder=nuplan_challenge \
        scenario_filter="$SPLIT" \
        worker.threads_per_node="$WORKERS_PER_JOB" \
        experiment_uid="$SPLIT/planTF" \
        verbose=false \
        planner.imitation_planner.planner_ckpt="$CKPT_ROOT/planTF.ckpt" \
        > "logs/${SPLIT}_${challenge}.log" 2>&1 &

done

wait
echo "[DONE] all challenges"


