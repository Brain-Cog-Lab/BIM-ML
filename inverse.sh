CUDA_VISIBLE_DEVICES=4 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 42&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42&
PID2=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method concat --seed 42&
PID3=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42&
PID4=$!;

CUDA_VISIBLE_DEVICES=7 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse&
PID5=$!;

CUDA_VISIBLE_DEVICES=6 python train_snn.py --model AVresnet18 --node-type ReLUNode --dataset CREMAD --epoch 100 --batch-size 64 --num-classes 6 --step 1 --modality audio-visual --alpha 0.8 --modulation Normal --fusion_method metamodal --meta_ratio 1.0 --seed 42 --inverse&