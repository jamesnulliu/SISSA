#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# echo ">>>>> SISSA-C >>>>>>"
# python test.py --model_config config/SISSA-C.yml --weights results/weights/SISSA-C/e-150_train-acc-0.99_val-acc-0.766.pt
# echo ">>>>> SISSA-C-A >>>>>>"
# python test.py --model_config config/SISSA-C-A.yml --weights results/weights/SISSA-C-A/e-120_train-acc-0.933_val-acc-0.766.pt
echo ">>>>> SISSA-R >>>>>>"
python test.py --model_config config/SISSA-R.yml --weights results/weights/SISSA-R/e-280_train-acc-0.913_val-acc-0.805.pt
echo ">>>>> SISSA-R-A >>>>>>"
python test.py --model_config config/SISSA-R-A.yml --weights results/weights/SISSA-R-A/e-380_train-acc-0.966_val-acc-0.804.pt
# echo ">>>>> SISSA-L >>>>>>"
# python test.py --model_config config/SISSA-L.yml --weights 
# echo ">>>>> SISSA-L-A >>>>>>"
# python test.py --model_config config/SISSA-L-A.yml --weights results/weights/SISSA-L-A/e-200_train-acc-1.0_val-acc-0.988.pt

echo "Done"