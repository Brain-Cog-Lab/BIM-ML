#--------1. CREMAD datasets-----------
CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --clip-grad 5.0&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --inverse --clip-grad 5.0&
PID2=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42 --clip-grad 5.0&
PID3=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse --clip-grad 5.0&
PID4=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --clip-grad 5.0&
PID5=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --inverse --clip-grad 5.0&
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42 --clip-grad 5.0&
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse --clip-grad 5.0&
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 2025&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 2025 --inverse&
#PID4=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025&
#PID5=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse&
#PID6=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 2025&
#PID7=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 2025 --inverse&
#PID8=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}


#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --seed 42 &
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --seed 42 --inverse &
#
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.3 --seed 42 &
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.3 --seed 42 --inverse &
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.5 --seed 42 &
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.5 --seed 42 --inverse &
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.7 --seed 42 &
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.7 --seed 42 --inverse &

#--------2. KineticSounds datasets-----------
CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --clip-grad 5.0&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42 --inverse --clip-grad 5.0&
PID2=$!;

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42 --clip-grad 5.0&
PID3=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse --clip-grad 5.0&
PID4=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --clip-grad 5.0&
PID5=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42 --inverse --clip-grad 5.0&
PID6=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42 --clip-grad 5.0&
PID7=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 64 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse --clip-grad 5.0&
PID8=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6} && wait ${PID7} && wait ${PID8}