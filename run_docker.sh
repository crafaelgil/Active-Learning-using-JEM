#!/bin/bash
sudo docker run \
--rm -it \
--shm-size=10g \
-v $(pwd):/workdir \
-w /workdir \
--gpus all \
pytorch/pytorch \
"$@"
#'"device=0"' 