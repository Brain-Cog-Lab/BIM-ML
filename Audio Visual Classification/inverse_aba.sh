CUDA_VISIBLE_DEVICES=0 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 0.1 --output ./exp_aba&
PID1=$!;

CUDA_VISIBLE_DEVICES=1 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 0.5 --output ./exp_aba&
PID2=$!;

CUDA_VISIBLE_DEVICES=2 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 1.0 --output ./exp_aba&
PID3=$!;

CUDA_VISIBLE_DEVICES=3 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 2.0 --output ./exp_aba&
PID4=$!;

CUDA_VISIBLE_DEVICES=4 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 3.0 --output ./exp_aba&
PID5=$!;

CUDA_VISIBLE_DEVICES=5 python train_aba.py --model AVresnet18 --node-type ReLUNode --dataset KineticSound --epoch 100 --batch-size 32 --num-classes 31 --step 1 --modality audio-visual --alpha 0.8 --modulation OGM_GE --fusion_method concat --seed 2025 --inverse --inverse-coef 5.0 --output ./exp_aba&
PID6=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4} && wait ${PID5} && wait ${PID6}