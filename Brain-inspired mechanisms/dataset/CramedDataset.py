import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb

import PIL
import torchaudio

class CramedDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = args.visual_path
        self.audio_feature_path = args.audio_path

        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.fps), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # audio
        # audio_filename = self.audio[idx]
        # spectrogram_path = os.path.dirname(audio_filename) + "-Spectrogram"
        # audio_name = os.path.basename(audio_filename).split('.')[0]
        # spectrogram_file = os.path.join(spectrogram_path, audio_name + '_spectrogram.npy')
        #
        # if os.path.exists(spectrogram_file):
        #     spectrogram = np.load(spectrogram_file)
        # else:
        #     samples, rate = librosa.load(audio_filename, sr=22050)
        #     resamples = np.tile(samples, 3)[:22050*3]
        #     resamples[resamples > 1.] = 1.
        #     resamples[resamples < -1.] = -1.
        #
        #     spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        #     spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        #
        #     # Save the spectrogram for later use
        #     np.save(spectrogram_file, spectrogram)

        audio_file_path = self.audio[idx]

        waveform, sample_rate = torchaudio.load(audio_file_path, normalize=True)
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=22050)
        waveform = torch.clamp(waveform, -1, 1)

        stft_transforms = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353, power=None, pad_mode='constant')
        spectrogram = stft_transforms(waveform)
        spectrogram = torch.log(torch.abs(spectrogram) + 1e-7)

        spectrogram = PIL.Image.fromarray(spectrogram.squeeze().numpy())  # (249, 257)

        spectrogram = transforms.Compose([
            transforms.Resize((224, 224)),  # 将频谱图像调整到224x224
            transforms.ToTensor(),
        ])(spectrogram)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.fps, 3, 224, 224))
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label