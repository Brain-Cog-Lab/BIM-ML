# 捕捉中断/终止信号，终止所有后台任务
trap 'echo "中断信号收到，正在终止所有子任务..."; pkill -P $$; exit 1' SIGINT SIGTERM

## ---------------------- seed 2025 ------------------------
## A. AVE-CI and K-S-CI in "LwF, SS-IL, AV-CIL" four method
## ----------1. LwF method
#pushd LwF
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 2025&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 2025&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 2025&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
## ----------2. SS-IL method ------------
#pushd SSIL
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
#
## -------------3. AV-CIL method -----
#pushd ours
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}


## ---------------------- seed 3917 ------------------------
## A. AVE-CI and K-S-CI in "LwF, SS-IL, AV-CIL" four method
## ----------1. LwF method
#pushd LwF
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 3917&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 3917&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 3917&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 3917&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
## ----------2. SS-IL method ------------
#pushd SSIL
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
#
## -------------3. AV-CIL method -----
#pushd ours
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}


## ---------------------- seed 0 ------------------------
## A. AVE-CI and K-S-CI in "LwF, SS-IL, AV-CIL" four method
# ----------1. LwF method
#pushd LwF
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 0&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 0&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 0&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_lwf.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 0&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
## ----------2. SS-IL method ------------
#pushd SSIL
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ssil.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --lr 1e-2 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
#
## -------------3. AV-CIL method -----
#pushd ours
#CUDA_VISIBLE_DEVICES=3 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ours.py --dataset AVE --num_classes 28 --class_num_per_step 7 --modality audio-visual --max_epoches 100 --num_workers 1 --memory_size 340 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
#PID2=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ours.py --dataset ksounds --num_classes 30 --class_num_per_step 6 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 30 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
#PID4=$!;
#
#popd
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}


## B. VGGSound-CI in "LwF, SS-IL, AV-CIL" four method
#pushd LwF
#CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 2025&
#PID2=$!;
#
#popd
#
#pushd SSIL
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID4=$!;
#popd

#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

#pushd ours
#CUDA_VISIBLE_DEVICES=4 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 2025&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 2025&
#PID2=$!;
#popd
#
#pushd ours
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID2=$!;
#popd
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
# B. VGGSound-CI in "LwF, SS-IL, AV-CIL" four method
#pushd LwF
#CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 3917&
#PID1=$!;
#
#CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 3917&
#PID2=$!;
#
#popd
#
#pushd SSIL
#CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 3917&
#PID3=$!;
#
#CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 3917&
#PID4=$!;
#popd
#
#wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}
#
#
## B. VGGSound-CI in "LwF, SS-IL, AV-CIL" four method
pushd LwF
CUDA_VISIBLE_DEVICES=4 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --seed 0&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_incremental_lwf.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --inverse --seed 0&
PID2=$!;

popd

pushd SSIL
CUDA_VISIBLE_DEVICES=6 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
PID3=$!;

CUDA_VISIBLE_DEVICES=7 python train_incremental_ssil.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
PID4=$!;
popd

wait ${PID1} && wait ${PID2} && wait ${PID3} && wait ${PID4}

pushd ours
CUDA_VISIBLE_DEVICES=4 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --seed 0&
PID1=$!;

CUDA_VISIBLE_DEVICES=5 python train_incremental_ours.py --dataset VGGSound_100 --num_classes 100 --class_num_per_step 10 --modality audio-visual --max_epoches 100 --num_workers 4 --memory_size 1500 --instance_contrastive --class_contrastive --instance_contrastive_temperature 0.05 --class_contrastive_temperature 0.05 --lam 0.5 --lam_I 0.1 --lam_C 1.0 --lr 1e-3 --lr_decay False --milestones 100 --weight_decay 1e-4 --train_batch_size 256 --infer_batch_size 128 --exemplar_batch_size 128 --inverse --seed 0&
PID2=$!;
popd