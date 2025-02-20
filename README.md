<div align="center">

<h2 class="papername"> Incorporating brain-inspired mechanisms for multisensory learning in artificial intelligence </h2>
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

\[[arxiv]()\] \[[paper]()\] \[[code](https://github.com/Brain-Cog-Lab/BIM-ML)\]

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FBrain-Cog-Lab%2FBIM-ML&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
<br>

</div>

Here is the PyTorch implementation of our paper. 
If you find this work useful for your research, please kindly cite our paper and star our repo.





## Datasets

- KineticsSound:

  We Follow We obtained the KineticsSound dataset using the method in [bmmal](https://github.com/MengShen0709/bmmal/tree/main) and thank them for their contribution! This will get a total of 31 classes, 14777 data for training and 2594 data for testing.

  1. Download dataset following [Kinetics Datasets Downloader](https://github.com/cvdfoundation/kinetics-dataset)
  2. Run **kinetics_convert_avi.py** to convert mp4 files into avi files.
  3. Run **kinetics_arrange_by_class.py** to organize the files.
  4. Run **extract_wav_and_frames.py** to extract wav files and 10 frame images as jpg.
  5. The final dataset file structure will be like:

```
├── kinetics_sound
    ├── my_train.txt
    ├── my_test.txt
    ├── train
        ├── video
            ├── label_name
                ├── vid_start_end
                    ├── frame_0.jpg
                    ├── frame_1.jpg
                    ├── ...
                    ├── frame_9.jpg
        ├── audio
            ├── label_name
                ├── vid_start_end.wav
                ├── ...
    ├── test
        ├── ...
```



In addition, [DMRNet](https://github.com/shicaiwei123/ECCV2024-DMRNet/tree/main) provides a version of the 33 categories, which can also be used as a reference, and their contribution is appreciated! 
