#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

# python data_gen.py

# echo ">>>>> SISSA-C >>>>>>"
# python train.py --model_config config/SISSA-C.yml
# echo "\n>>>>> SISSA-C-A >>>>>>"
# python train.py --model_config config/SISSA-C-A.yml
echo "\n>>>>> SISSA-R >>>>>>"
python train.py --model_config config/SISSA-R.yml
echo "\n>>>>> SISSA-R-A >>>>>>"
python train.py --model_config config/SISSA-R-A.yml
# echo "\n>>>>> SISSA-L >>>>>>"
# python train.py --model_config config/SISSA-L.yml
# echo "\n>>>>> SISSA-L-A >>>>>>"
# python train.py --model_config config/SISSA-L-A.yml

echo "Done"