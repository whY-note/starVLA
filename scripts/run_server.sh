source /home/smartmore/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path /data1t/haowenYan/Checkpoints/0730__qwengr00t/checkpoints/steps_100000_pytorch_model.pt \
    --port 5694 \
    --use_bf16