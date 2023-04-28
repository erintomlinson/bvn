#!/usr/bin/env python3
import os
import sys
import re
import subprocess
import numpy as np
from dataclasses import dataclass, fields
from itertools import product

submit_jobs = True

@dataclass
class Config:
    env_name: str = 'FetchReach-v1'
    n_workers: int = 2
    n_epochs: int = 50
    seed: int = 100
    metric_embed_dim: int = 16
    partition: str = 'seas_gpu'
    constraint: str = 'skylake'
    n_cpus: int = 1

# Setup sweep parameters like in the original train.py script
env_name_ = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1']
n_workers_ = [2, 8, 16, 20]
n_epochs_ = [50, 150, 200, 500]
n_cpus_ = [1, 2, 4, 8]
seed_ = [100, 200, 300, 400, 500]

exps = []
for (env_name, n_workers, n_epochs, n_cpus), seed in product(zip(env_name_, n_workers_, n_epochs_, n_cpus_), seed_):
    exp = Config(env_name=env_name, n_workers=n_workers, n_epochs=n_epochs, n_cpus=n_cpus, seed=seed)
    exps.append(exp)

# Update to match your fetch folder and python locations
fetch = '/n/holyscratch01/protopapas_lab/Everyone/etomlinson/repos/bvn/fetch'
python = '/n/home01/etomlinson/.conda/envs/bvn/bin/python3'
train_template = f'{fetch}/experiments/bvn/train_template.py'

if not os.path.exists('train_logs'):
   os.makedirs('train_logs')

for idx, exp in enumerate(exps):

    # Create train script specific for this job using train_template.py
    with open(train_template, 'r') as f:
        template = f.read()

    for field in fields(exp):
        value = getattr(exp, field.name)
        template = template.replace(f'<{field.name.upper()}>', f"'{value}'" if isinstance(value, str) else f'{value}')

    train_script = f'{fetch}/experiments/bvn/train_{idx}.py'
    with open(train_script, 'w') as f:
        f.write(template)

    # Build SLURM job submission script
    job_name = f'fetch_sweep_{idx}'
    queue_file = f'{job_name}.queue'
    with open(queue_file, 'w') as f:

        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --job-name={job_name}\n')
        f.write(f'#SBATCH --output train_logs/{job_name}_%j.out\n')
        f.write(f'#SBATCH --error train_logs/{job_name}_%j.err\n')
        f.write(f'#SBATCH --partition={exp.partition}\n')
        if exp.constraint: 
            f.write(f'#SBATCH --constraint={exp.constraint}\n')
        f.write(f'#SBATCH --time 5-00:00\n')
        f.write(f'#SBATCH --cpus-per-task={n_cpus}\n')
        f.write(f'#SBATCH --gres=gpu:1\n')
        f.write(f'#SBATCH --mem=16G\n')
        f.write(f'\n')
        f.write(f'module load python/3.8.5-fasrc01\n')
        f.write(f'module load gcc/12.1.0-fasrc01\n')
        f.write(f'module load intel/21.2.0-fasrc01\n')
        f.write(f'module load openmpi/4.1.3-fasrc01\n')
        f.write(f'module load cuda/11.7.1-fasrc01\n')
        f.write(f'source activate bvn\n')
        f.write(f'\n')
        f.write(f'export MPICC=$(which mpicc)\n')
        f.write(f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/home01/etomlinson/.mujoco/mujoco210/bin\n')
        f.write(f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia\n')
        f.write(f'\n')
        f.write(f'cd {fetch}\n')
        f.write(f'export PYTHONPATH=$PYTHONPATH:$(pwd)\n')
        f.write(f'export ML_LOGGER_ROOT=$(pwd)/results\n')
        f.write(f'{python} experiments/bvn/train_{idx}.py\n')

    if submit_jobs:

        sbatch = subprocess.run(f'sbatch {queue_file}'.split(), check=True, stdout=subprocess.PIPE, universal_newlines=True, timeout=15)
        match = re.search('\d+', sbatch.stdout)
        if match:
            print(f'Submitted {job_name} and received JobID {match.group()}.')
        else:
            print(f'Warning: {job_name} was submitted but no JobID was returned by SLURM')
