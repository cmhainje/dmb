#!/bin/bash
#SBATCH -J ${name}
#SBATCH -t 2-00:00:00
#SBATCH --mem 40G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 16
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=all
#SBATCH --mail-user=ch4407@nyu.edu
#SBATCH --account torch_pr_477_general
#SBATCH --output=${directory}/slurm-%x-%j.out
#SBATCH --error=${directory}/slurm-%x-%j.out

module purge
apptainer exec \
  --env LD_LIBRARY_PATH=/home/ch4407/local/lib:$$LD_LIBRARY_PATH \
  /share/apps/images/ubuntu-24.04.3.sif \
  mpiexec -bind-to none -np 16 \
  ${exepath} \
  ${directory}/gizmo.param
