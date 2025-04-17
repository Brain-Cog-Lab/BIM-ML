# ----------inverse ---------------
#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse &
#PID4=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

#CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 &
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse &
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 &
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse &
#PID4=$!;
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 &
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse &
PID2=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 &
PID3=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type LIFNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 4 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse &
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}