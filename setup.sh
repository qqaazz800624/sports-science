srun --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=4 \
     --partition=defq  \
     --gres=gpu:pro6000:1 \
     --cpu-bind=none \
     --time=01:00:00 \
     --container-image='/data/container-images/enroot/nvidia+pytorch+25.03-py3.sqsh' \
     --container-mount-home --pty /bin/bash -l