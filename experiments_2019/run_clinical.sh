#!/bin/bash

set -e

if [ -z "$CONFIG_NAME" ] || [ -z "$EXEC_ID" ] || [ -z "$NUM_FEATURES" ]; then
    echo "❌ Missing required environment variables."
    exit 1
fi

echo "✅ Starting training for: $CONFIG_NAME | Run: $EXEC_ID | Features: $NUM_FEATURES"

EXPERIMENT_BASE="/scratch/tsaraujo/saude/hcpa"
CONFIG_DIR=experiments_2019/$"$CONFIG_NAME"
TFREC_PATH="$CONFIG_DIR/tfrec"
FINAL_RESULTS="/home/users/tsaraujo/clinical_data/experiments_2019/$NUM_FEATURES/$CONFIG_NAME/$EXEC_ID"

cd "$EXPERIMENT_BASE"

CUDA_VISIBLE_DEVICES=0,1,2 python3 dr_hcpa_v2_2024.py \
    --tfrec_dir "$TFREC_PATH" \
    --dataset hcpa \
    --results results \
    --epochs 200 \
    --batch_size 8 \
    --exec "$EXEC_ID" \
    --verbose 0

mkdir -p "$FINAL_RESULTS"

echo "📦 Copying results to $FINAL_RESULTS"
cd results
cp -r * "$FINAL_RESULTS"
rm -rf *

echo "✅ Finished training for $CONFIG_NAME | Run $EXEC_ID"
