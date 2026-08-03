echo "Evaluating trajectory actions on training data offline ..."

cd /dataset/vhaowenyan/projects/starVLA

source /dataset/vhaowenyan/miniforge3/etc/profile.d/conda.sh

conda activate starVLA

python tools/offline_eval_traj_actions.py \
  --checkpoint results/Checkpoints/0730__qwengr00t/checkpoints/steps_100000_pytorch_model.pt \
  --benchmark-name DualFranka \
  --trajectory-index 1 \
  --use-bf16