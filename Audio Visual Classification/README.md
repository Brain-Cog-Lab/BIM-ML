Code for **Audio Visual Classification task**



## Training Script

1. Overall Performance: [inverse_ann.sh](./inverse_ann.sh), [inverse_snn.sh](inverse_snn.sh)
2. Mechanism Ablation: [inverse_aba.sh](./inverse_aba.sh) & [inverse_joint.sh](./inverse_joint.sh)
3. Finetuning: [inverse_joint.sh](./inverse_joint.sh)



## Figure for paper

You can draw all the diagrams in Figure 2 of the paper from the [plot.py](./plot.py) file.



## Datasets

- CREMA-D:

CREMA-D datasets：[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

- Kinetics-Sounds:

  We Follow We obtained the KineticsSound dataset using the method in [bmmal](https://github.com/MengShen0709/bmmal/tree/main) and thank them for their contribution! This will get a total of 31 classes, 14772 data for training and 2594 data for testing. You can download our zipped dataset [here]().

  1. Download dataset following [Kinetics Datasets Downloader](https://github.com/cvdfoundation/kinetics-dataset)
  2. Run **kinetics_convert_avi.py** to convert mp4 files into avi files.
  3. Run **kinetics_arrange_by_class.py** to organize the files.
  4. Run **extract_wav_and_frames.py** to extract wav files and 10 frame images as jpg.
  5. Run **ks_find.py** to filter out disqualified data.
  6. The final dataset file structure will be like:

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

- UrbanSound8K-AV:

UrbanSound8K-AV datasets: [UrbanSound8K-AV](https://github.com/Guo-Lingyue/SMMT)



## Vanilla Methods

[OGM_GE](https://github.com/GeWu-Lab/OGM-GE_CVPR2022), MSLR, [LFM](https://github.com/njustkmg/NeurIPS24-LFM)



## Results

| Methods | CRMEA-D |  |  |  | Kinetics-Sounds |  |  |  | UrbanSound8K-AV |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Normal | MSLR | OGM_GE | LFM | Normal | MSLR | OGM_GE | LFM | Normal | MSLR | OGM_GE | LFM |
| Vanilla | 62.63 | 64.11 | 68.68 | 64.11 | 51.58 | 51.89 | 57.63 | 55.28 | 97.90 | 97.79 | 97.60 | 98.05 |
| w/ inverse | 63.44 | 65.59 | 71.10 | 63.98 | 56.17 | 55.86 | 64.61 | 63.15 | 97.86 | 97.98 | 99.24 | 98.63 |

