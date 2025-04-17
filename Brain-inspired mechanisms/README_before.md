Code for 《Biologically Inspired Metamodal Spiking Neural Networks: for Efficient Multimodal Perceptual Learning》



# Training Script

## 1. Baseline

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation Normal --fusion_method concat
```



## 2. OGM_GE

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat
```



## 3. Inverse

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation OGM_GE --fusion_method concat --inverse
```



## 4. Metamodal

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1
```



## 5. Metamodal without meta

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.0 --inverse
```



## 6. Inverse + Metamodal

```bash
CUDA_VISIBLE_DEVICES='5' python main.py --alpha 0.8 --ckpt_path result_test/ --train --batch_size 64 --modulation OGM_GE --fusion_method metamodal --meta_ratio 0.1 --inverse
```



## 7. Unimodal_Finetune

```bash
CUDA_VISIBLE_DEVICES='5' python main_unimodalFintune.py --train --ckpt_path result_unimodal/ --model_path /home/hexiang/OGM-GE_CVPR2022/result/Normal_inverse_False_alpha_0.8_bs_64_fusion_concat_metaratio_-1.0_epoch_75_acc_0.6061827956989247_seed_0.pth  # 1

CUDA_VISIBLE_DEVICES='5' python main_unimodalFintune.py --train --ckpt_path result_unimodal/ --model_path /home/hexiang/OGM-GE_CVPR2022/result/OGM_GE_inverse_True_alpha_0.8_bs_64_fusion_ogmge_metamodal_metaratio_0.1_epoch_81_acc_0.6774193548387096_seed_0.pth  # 2
```



部分代码copy from：

https://github.com/GeWu-Lab/OGM-GE_CVPR2022
