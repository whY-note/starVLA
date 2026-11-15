# B3 双臂叠积木训练

使用 `train_starvla.py`、QwenOFT 和单机八卡 DeepSpeed ZeRO-2，接入已有的
`data_v5_eef_lerobot_v21` 数据集。

## 数据与动作语义

默认数据父目录为 `/dataset-v2/haowenyan/project/data/MRUI-Dataset/stack_blocks_latest`。
103 个 episode、150,169 帧、50 Hz；三路相机顺序固定为正面、左腕、右腕。
状态和动作顺序均为：左位置 3、左四元数 4、左夹爪 1、右位置 3、右四元数 4、右夹爪 1。
位置单位为米，四元数保持源数据的 `xyzw` 顺序；动作是基座坐标系下的绝对目标。

动作窗口为 `[t, ..., t+15]`，状态与图像取 `t`。轨迹末尾重复最后一个绝对目标进行填充。
状态和动作均使用各字段的 `min_max` 统计归一化到 `[-1, 1]`，两个夹爪保留连续开度。
没有旋转表示转换。部署时应使用同一统计做 min/max 反归一化，保持 `xyzw` 顺序，
并将预测四元数归一化为单位四元数；夹爪恢复到 `[0, 1]` 开度语义。

当前 recipe 使用全部 103 条轨迹。训练器的周期动作误差检查使用训练批次，
不代表独立验证集的泛化性能。若需要验证集，应按完整 episode 划分并单独计算训练集统计。

## 检查数据

在仓库根目录执行：

```bash
NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python \
  examples/realRobots/B3/train_files/check_dataset.py \
  --install-modality --full-scan --num-workers 2
```

检查器将仓库中的 `train_files/modality.json` 安装到数据的 `meta/modality.json`；
已有映射不一致时会报错。加载器会在数据的 `meta/` 中生成统计和步索引缓存。
报告和归一化统计保存到 `results/B3/data_check/`。
检查包含全部文件与行数、维度和字段顺序、完整数值扫描、三视角解码、动作窗口与末尾填充、
归一化往返一致性，以及训练使用的 DataLoader/collator。

可通过 `--datasets.vla_data.data_root_dir /另一个数据父目录` 覆盖路径。
有 CUDA 和已准备好的模型权重时，可附加 `--model-smoke` 检查模型前向、反向和推理。

## 八卡启动

使用仓库 `.venv`；也可设置 `PYTHON_BIN` 指向装好 starVLA 依赖的 Python。
默认骨干为 `Qwen/Qwen3-VL-4B-Instruct`。离线机器先准备权重，并用 `BASE_VLM` 指向本地目录；
目录名应包含 `Qwen3-VL`，以匹配当前仓库的 VLM 路由规则。

先运行 10 步，验证八卡同步和保存：

```bash
BASE_VLM=/path/to/Qwen3-VL-4B-Instruct \
  bash examples/realRobots/B3/train_files/run_train_8gpu.sh \
  --run_id b3_stack_blocks_smoke \
  --trainer.max_train_steps 10 \
  --trainer.num_warmup_steps 1 \
  --trainer.save_interval 5 \
  --trainer.eval_interval 5 \
  --trainer.logging_frequency 1
```

正式训练：

```bash
BASE_VLM=/path/to/Qwen3-VL-4B-Instruct \
  bash examples/realRobots/B3/train_files/run_train_8gpu.sh
```

默认每卡 batch 为 2、梯度累积为 1，全局 batch 为 16；初始训练长度为 10,000 步。
显存不足时可追加 `--datasets.vla_data.per_device_batch_size 1`。
累积步数与现有 DeepSpeed JSON 固定为 1，检查器会拒绝仅修改训练 YAML 的累积配置。
脚本先在单进程中检查数据并准备缓存，再启动八个进程；`--check-only` 只运行数据检查。
默认关闭 W&B，可通过 `WANDB_MODE=online` 和相应 YAML/CLI 设置启用。

输出目录为 `/dev/shm/haowenyan/checkpoints/<run_id>/`，包含配置、数据统计、周期模型权重与最终权重。
默认 `run_id` 为 `b3_stack_blocks_qwenoft`；每个任务目录内，周期权重位于 `checkpoints/`，
最终权重位于 `final_model/pytorch_model.pt`。
现有训练器的保存逻辑主要保存模型参数；精确恢复优化器和随机数状态需要另行扩展。
