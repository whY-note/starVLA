echo "train starVLA with zip tie task for dual Franka robot"

cd /dataset/vhaowenyan/projects/starVLA

export PATH=/dataset/vhaowenyan/miniforge3/bin:$PATH

source /dataset/vhaowenyan/miniforge3/etc/profile.d/conda.sh

conda activate starVLA

nvidia-smi

bash examples/realRobots/Franka/train_files/run_franka_train_dual.sh