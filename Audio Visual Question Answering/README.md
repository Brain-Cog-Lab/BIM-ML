Code for **Audio Visual Question Answering**



## Training Script

The run script using the IEMF method is as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python net_grd_avst\main_avst.py --mode train --inverse
```



## Figure for paper

You can draw all the diagrams in Figure 4 of the paper from the [plot.py](./plot.py) file.



## Datasets

- Please organize the dataset in the way as [MUSIC_AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA). You can also download our well uploaded zip archive from [here]()! 



## Results



| Method             | Audio Question (%) |             |       | Visual Question (%) |          |       | Audio-Visual Question (%) |          |          |             |          |       | Overall Avg. (%) |
| :----------------- | :----------------- | :---------- | :---- | :------------------ | :------- | :---- | :------------------------ | :------- | :------- | :---------- | :------- | :---- | :--------------- |
|                    | Counting           | Comparative | Avg   | Counting            | Location | Avg   | Existential               | Location | Counting | Comparative | Temporal | Avg   |                  |
| Baseline           | 77.20              | 62.06       | 71.60 | 74.15               | 76.79    | 75.48 | 81.71                     | 67.43    | 62.57    | 61.61       | 62.99    | 67.34 | 70.24            |
| Baseline w/ IEMF   | 77.40              | 63.89       | 72.40 | 75.23               | 76.79    | 76.02 | 82.11                     | 69.07    | 59.55    | 62.87       | 64.93    | 67.87 | 70.82            |
| ST-AVQA            | 77.59              | 62.23       | 71.90 | 73.89               | 75.57    | 74.74 | 82.81                     | 68.45    | 63.00    | 60.45       | 62.86    | 67.61 | 70.26            |
| ST-AVQA w/ inverse | 79.84              | 65.39       | 74.49 | 74.65               | 76.63    | 75.65 | 82.11                     | 69.47    | 62.68    | 62.51       | 64.20    | 68.33 | 71.36            |
| ST-AVQA w/ joint   | 78.18              | 63.39       | 72.70 | 75.06               | 75.98    | 75.53 | 82.91                     | 68.84    | 63.00    | 61.52       | 61.89    | 67.81 | 70.71            |





## Acknowledgements

The code is based on [MUSIC_AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA), thanks for their work!
