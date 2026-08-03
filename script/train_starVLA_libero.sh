echo "train starVLA with libero"

cd /dataset/vhaowenyan/projects/starVLA

export PATH=/dataset/vhaowenyan/miniforge3/bin:$PATH

source /dataset/vhaowenyan/miniforge3/etc/profile.d/conda.sh

conda activate starVLA

nvidia-smi

python -c "import flash_attn; print(flash_attn.__version__)"

bash examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh