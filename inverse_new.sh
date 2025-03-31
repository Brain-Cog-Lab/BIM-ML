#--------1. CREMAD datasets-----------
CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 &
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --inverse &
PID2=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 42 &
PID3=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 42 --inverse &
PID4=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 &
PID5=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --inverse &
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 42 &
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 42 --inverse &
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse  --psai 2.0&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse  --psai 0.1&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse  --psai 2.0&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse  --psai 0.1&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse  --psai 2.0&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse  --psai 0.1&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse  --psai 2.0&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse  --psai 0.1&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}

##--------2. KineticSounds datasets-----------
CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --inverse&
PID2=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 42&
PID3=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 42 --inverse&
PID4=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42&
PID5=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --inverse&
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 42&
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 42 --inverse&
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --psai 0.1&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --psai 0.5&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse --psai 0.1&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method avattn --seed 2025 --inverse --psai 0.5&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse --psai 0.1&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse --psai 0.5&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse --psai 0.1&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method avattn --seed 2025 --inverse --psai 0.5&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}