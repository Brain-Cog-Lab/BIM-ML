# 捕捉中断/终止信号，终止所有后台任务
trap 'echo "中断信号收到，正在终止所有子任务..."; pkill -P $$; exit 1' SIGINT SIGTERM

##--------1. CREMAD datasets-----------
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse_starts 60&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse --inverse_starts 60&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}


CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025 &
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 32 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025 --inverse --inverse_starts 60&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 16 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --lr 2e-3 --seed 2025 &
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 16 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --lr 2e-3 --seed 2025 --inverse --inverse_starts 60&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

##--------2. KineticSounds datasets-----------
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025 --inverse&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 16 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --seed 2025&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 16 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --seed 2025 --inverse&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

##--------3. UrbanSound8K-AV datasets-----------
CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 2025 --inverse&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

CUDA_VISIBLE_DEVICES=0 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 32 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation MSLR --fusion_method concat --seed 2025 --inverse&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 16 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --seed 2025&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset UrbanSound8K --epoch 100 --batch-size 16 --num-classes 10 --step 1 --modality audio-visual --alpha 0.8 --modulation LFM --fusion_method concat --seed 2025 --inverse&
PID4=$!;

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}