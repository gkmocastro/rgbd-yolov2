#!/bin/bash
#SBATCH --job-name='train_rgbd_anything'
#SBATCH --chdir=/home/escorpiao/workspace/depth-obj-detection/rgbd-yolov2
#SBATCH --output=/home/escorpiao/workspace/depth-obj-detection/rgbd-yolov2/logs/log.out
#SBATCH --error=/home/escorpiao/workspace/depth-obj-detection/rgbd-yolov2/logs/log.out
#SBATCH --partition=gpu
#SBATCH --time=0-24:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

export SLURM_OVERLAP=yes


source '/opt/intel/oneapi/setvars.sh' 
export CONDA_ENV=pytorch-gpu 



module try-load singularity
echo job start time is `date`


exec_node=${SLURM_JOB_NODELIST}
cpu_curr_node=${SLURM_CPUS_ON_NODE}
if [ "${cpu_curr_node}" = "" ]; then  cpu_curr_node={{ cores_per_node|default:1 }}; fi

set -x
srun -N1 -n1 --cpu_bind=cores -l --nodelist=${exec_node} \
    --cpus-per-task=${cpu_curr_node} \
    singularity exec --nv \
    -B /home/escorpiao -B /scratch \
    --pwd /home/escorpiao/workspace/depth-obj-detection/rgbd-yolov2 \
    /home/escorpiao/.lico/container/allPython.sif bash /home/escorpiao/workspace/depth-obj-detection/rgbd-yolov2/train_rgbd.sh 
set +x


echo job end time is `date`
