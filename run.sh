CUDA_VISIBLE_DEVICES=0 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation Normal --fusion_method concat --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=0 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation Normal --fusion_method concat --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=0 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation Normal --fusion_method concat --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}


CUDA_VISIBLE_DEVICES=1 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=1 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=1 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}


CUDA_VISIBLE_DEVICES=2 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --inverse --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=2 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --inverse --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=2 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --inverse --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}


CUDA_VISIBLE_DEVICES=3 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=3 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=3 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}

CUDA_VISIBLE_DEVICES=4 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.0 --inverse --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=4 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.0 --inverse --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=4 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.0 --inverse --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}


CUDA_VISIBLE_DEVICES=5 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --inverse --seed 0&
PID1=$!;
CUDA_VISIBLE_DEVICES=5 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --inverse --seed 42&
PID2=$!;
CUDA_VISIBLE_DEVICES=5 python main.py --alpha 0.8 --ckpt_path result/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --inverse --seed 2025&
PID3=$!;
#wait ${PID1} && wait ${PID2} && wait ${PID3}