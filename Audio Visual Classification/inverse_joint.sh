# ------------ 1. 没有inverse-------------
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --output ./exp_joint&
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse --output ./exp_joint&
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025 --inverse --output ./exp_joint&
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --seed 2025 --inverse --output ./exp_joint&


# ------------2. 微调 --------------
CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio --alpha 0.8 --seed 2025 &

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality visual --alpha 0.8 --seed 2025 &

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio --alpha 0.8 --seed 2025 --load-avmodel --output ./exp_finetuning &

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio --alpha 0.8 --seed 2025 --load-avmodel --inverse --output ./exp_finetuning &

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality visual --alpha 0.8 --seed 2025 --load-avmodel --output ./exp_finetuning &

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality visual --alpha 0.8 --seed 2025 --load-avmodel --inverse --output ./exp_finetuning &


##CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio --alpha 0.8 --seed 2025 &
#
#CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality visual --alpha 0.8 --seed 2025 &
#
##CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio --alpha 0.8 --seed 2025 --load-avmodel --output ./exp_finetuning &
#
##CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio --alpha 0.8 --seed 2025 --load-avmodel --inverse --output ./exp_finetuning &
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality visual --alpha 0.8 --seed 2025 --load-avmodel --output ./exp_finetuning &
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality visual --alpha 0.8 --seed 2025 --load-avmodel --inverse --output ./exp_finetuning &