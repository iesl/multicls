#!/bin/bash

RANK=0
WORLD_SIZE=1
CUDA_LAUNCH_BLOCKING=1

#~/anaconda3/bin/fil-profile run ./olfmlm/pretrain_bert.py "$@"
~/anaconda3/bin/python -m olfmlm.pretrain_bert "$@"
#~/anaconda3/bin/mprof run ~/anaconda3/bin/python -m olfmlm.pretrain_bert "$@"
#~/anaconda3/bin/python -m memory_profiler ./olfmlm/pretrain_bert.py "$@"
