#!/bin/bash
DATA_DIR=/disk/vanishing_data/kx205/RVT1/gen1/

CKPT_PATH=/disk/vanishing_data/kx205/RVT1/rvt_model/rvt-b.ckpt

USE_TEST=1 # evaluate on the validation set

GPU_ID=0

MDL_CFG=base

python Huang_visualization.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=${USE_TEST} hardware.gpus=${GPU_ID} +experiment/gen1="${MDL_CFG}.yaml" \
batch_size.eval=2 model.postprocess.confidence_threshold=0.001
