# --------联合训练测试------------------
##--------1. CREMAD datasets-----------
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --output ./exp_joint --tensorboard-dir ./exp_joint&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --output ./exp_joint --tensorboard-dir ./exp_joint&
#PID2=$!;
#
#
###--------2. KineticSounds datasets-----------
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --output ./exp_joint --tensorboard-dir ./exp_joint&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --output ./exp_joint --tensorboard-dir ./exp_joint&
#PID4=$!;


# ----------inverse ---------------
CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse &
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse &
PID2=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse &
PID3=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse &
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse &
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse &
PID2=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse &
PID3=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse &
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}