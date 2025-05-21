<div align="center">
<h2 class="papername"> Incorporating brain-inspired mechanisms for multimodal learning in artificial intelligence </h2>
<div>
<div>
    <a href="https://scholar.google.com/citations?user=Em5FqXYAAAAJ" target="_blank">Xiang He*</a>,
    <a href="https://scholar.google.com/citations?user=2E9Drq8AAAAJ" target="_blank">Dongcheng Zhao*</a>,
    <a href="https://scholar.google.com/citations?user=3QpRLTgAAAAJ" target="_blank">Yang Li</a>,
    <a href="https://scholar.google.com/citations?user=Sv-WdBkAAAAJ" target="_blank">Qingqun Kong†</a>,
    <a href="https://ieeexplore.ieee.org/author/37085719247" target="_blank">Xin Yang†</a>,
    <a href="https://scholar.google.com/citations?user=Rl-YqPEAAAAJ" target="_blank">Yi Zeng†</a>
</div>
Institute of Automation, Chinese Academy of Sciences, Beijing<br>
*Equal contribution
†Corresponding author

\[[arxiv](https://arxiv.org/abs/2505.10176)\] \[[paper]()\] \[[code](https://github.com/Brain-Cog-Lab/IEMF)\]

</div>
<br>

</div>

Here is the PyTorch implementation of our paper. 
If you find this work useful for your research, please kindly cite our paper and star our repo.



## Method

We propose an inverse effectiveness driven multimodal fusion (IEMF) method, which dynamically adjusts the update dynamics of the multimodal fusion module based on the relationship between the strength of individual modality cues and the strength of the fused multimodal signal.

![](./figs/method.jpg)



## Usage

```bash
+--- Audio Visual Classification 
+--- Audio Visual Continual Learning
\--- Audio Visual Question Answering
```

Three folders provide three tasks each. They contain detailed **run scripts for each task, drawing programs, and the way to download the corresponding dataset.** 

## Well-trained model
We also upload the weights of the trained model, as well as the log files from the training process here to ensure reproduction of the results in the paper. You can find them at [https://huggingface.co/xianghe/IEMF/tree/main](https://huggingface.co/xianghe/IEMF/tree/main).

## Dataset Download
You can find how to download the dataset under the folder corresponding to each task. 
In particular, due to the processing complexity of the Kinetics-Sounds dataset, you can download our packaged raw video-audio dataset at [here](https://pan.baidu.com/s/1NHmpyhpPaXJVgtwFPkKHcw) (extraction code:  bauh).
In addition to the original dataset, we also provide processed data in HDF5 format ready for network model input, which you can access [here](https://pan.baidu.com/s/1v28Pt9HUKHUv8JCagdGuTQ) (extraction code: jzbg).

## Citation

If our paper is useful for your research, please consider citing it:

```bash
@misc{he2025incorporatingbraininspiredmechanismsmultimodal,
      title={Incorporating brain-inspired mechanisms for multimodal learning in artificial intelligence}, 
      author={Xiang He and Dongcheng Zhao and Yang Li and Qingqun Kong and Xin Yang and Yi Zeng},
      year={2025},
      eprint={2505.10176},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2505.10176}, 
}
```





## Acknowledgements

The code for each of the three tasks refers to [OGM_GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022), [AV-CIL_ICCV2023](https://github.com/weiguoPian/AV-CIL_ICCV2023), [MUSIC_AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA):, thanks for their excellent work!

If you are confused about using it or have other feedback and comments, please feel free to contact us via hexiang2021@ia.ac.cn. Have a good day!
