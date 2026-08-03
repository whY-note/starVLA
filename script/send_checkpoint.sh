#!/usr/bin/env bash
set -euo pipefail

target_user="smartmore"
target_ip="10.80.185.41"
source_dir="/dataset/vhaowenyan/projects/starVLA/results/Checkpoints/0730__qwengr00t"
target_dir="/data1t/haowenYan/Checkpoints/0730__qwengr00t"

# 查看 checkpoint 大小
du -sh "${source_dir}/checkpoints/steps_100000_pytorch_model.pt"

# 在一次 SSH 连接中创建目录并同步所有必需文件。
# /./ 是 rsync 的相对路径起点，用于在远端保留 checkpoints/ 层级。
rsync -aHR --info=progress2 --partial \
  --rsync-path="mkdir -p '${target_dir}' && rsync" \
  "${source_dir}/./checkpoints/steps_100000_pytorch_model.pt" \
  "${source_dir}/./config.yaml" \
  "${source_dir}/./dataset_statistics.json" \
  "${target_user}@${target_ip}:${target_dir}/"
