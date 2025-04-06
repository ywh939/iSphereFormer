#!/bin/bash

# 设置 CUDA_VISIBLE_DEVICES 来选择使用的 GPU
export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 GPU 1

# 获取 Docker 内部 IP 地址，确保 MASTER_ADDR 可解析
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500  # 设置通信端口

# 启动 torchrun 进行 DDP 训练
torchrun --nnodes=1 --nproc_per_node=2 train_ddp.py --config config/semantic_kitti/semantic_kitti_ms_25.yaml
# torchrun --nnodes=1 --nproc_per_node=2 train_ddp.py --config config/semantic_kitti/semantci_kitti_mamba.yaml
