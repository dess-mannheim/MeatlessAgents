#!/bin/bash
#SBATCH --job-name=HealthyAgents
#SBATCH --output=slurm_logs/%j_out.log
#SBATCH --error=slurm_logs/%j_err.log
##SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G

source /opt/anaconda3/etc/profile.d/conda.sh

conda activate healthy_gabm

# for 2 GPUs, add --tensor-parallel-size 2
# for quantization, add --quantization bitsandbytes --load-format bitsandbytes
vllm serve meta-llama/Llama-3.2-3B-Instruct --api-key token-abc123 --task generate --max_model_len 30000 --port 8080 --tensor-parallel-size 2 --enable-prefix-caching > ./slurm_logs/vllm_2.log &
#vllm serve meta-llama/Llama-3.1-8B-Instruct --api-key token-abc123 --task generate --max_model_len 30000 --port 8080 --tensor-parallel-size 2 --enable-prefix-caching > ./slurm_logs/vllm_2.log &
#vllm serve meta-llama/Llama-3.3-70B-Instruct --api-key token-abc123 --task generate --max_model_len 15000 --port 8080 --tensor-parallel-size 2 --enable-prefix-caching > ./slurm_logs/vllm_2.log &
export GENERATE_BASE_URL=http://localhost:8080/v1
export GENERATE_API_KEY=token-abc123

# Wait for vllm to start
sleep 120

python gabm_simulation.py