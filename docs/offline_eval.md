# Offline Evaluation 离线轨迹评估

## 输出路径

输出路径自动生成：
```
./results/offline_eval/
└── {ckpt_name}_{steps}/
    └── {benchmark_name}/
        └── episode_{trajectory_id}/
            ├── action_comparison.png
            ├── actions.csv
            └── metrics.json
```

例如：
```
results/offline_eval/
└── 0803_franka_dual_zip_tie_qwengr00t_100000/
    └── Franka/
        └── episode_0/
```

## 使用方法
Franka 双臂示例：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint results/Checkpoints/0803_franka_dual_zip_tie_qwengr00t/checkpoints/steps_100000_pytorch_model.pt \
  --benchmark-name Franka \
  --trajectory-index 0 \
  --use-bf16
```

先测试前 10 帧：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint /path/to/steps_100000_pytorch_model.pt \
  --benchmark-name Franka \
  --trajectory-index 0 \
  --max-frames 10 \
  --use-bf16
```
LIBERO 示例：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint /path/to/libero_run/checkpoints/steps_50000_pytorch_model.pt \
  --benchmark-name LIBERO \
  --trajectory-id 3 \
  --use-bf16
  ```

数据路径发生变化时：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint /path/to/checkpoint.pt \
  --benchmark-name RoboCasa \
  --data-root-dir /path/to/lerobot/datasets \
  --data-mix robocasa_mix_name \
  --trajectory-index 0 \
  --use-bf16
```


对于需要特殊推理参数的扩散模型：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint /path/to/checkpoint.pt \
  --benchmark-name Franka \
  --trajectory-index 0 \
  --predict-arg use_ddim=true \
  --predict-arg num_ddim_steps=20 \
  --use-bf16
```


对于多数据集 checkpoint：

```bash
python tools/offline_eval_traj_actions.py \
  --checkpoint /path/to/checkpoint.pt \
  --benchmark-name OXE \
  --dataset-index 1 \
  --unnorm-key oxe_rt1 \
  --trajectory-index 0 \
  --use-bf16
```