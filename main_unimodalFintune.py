import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pdb

from dataset.CramedDataset import CramedDataset
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
import re
import math

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='identical', type=str,
                        choices=['sum', 'concat', 'gated', 'film', 'identical', 'metamodal'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/home/hexiang/data/datasets/CREMA-D/AudioWAV/', type=str)
    parser.add_argument('--visual_path', default='/home/hexiang/data/datasets/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', type=str, help='path to save trained models', default='')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--meta_ratio', default=-1., type=float, help='meta ratio')

    parser.add_argument('--model_path', type=str, help='path to save trained models', default='')

    # snr value
    parser.add_argument('--snr', default=0, type=int,
                        help='random noise amplitude controled by snr, 0 means no noise')

    parser.add_argument('--snrModality', type=str, help='which Modality')
    return parser.parse_args()

args = get_arguments()

modulation, inverse, bs, meta_ratio, fusion, seed, alpha = None, None, None, None, None, None, None

if len(args.model_path) > 0:
    modulation = re.search(r'([^/]+)_inverse', args.model_path).group(1)
    inverse = re.search(r'inverse_(.*?)_', args.model_path).group(1)
    bs = re.search(r'bs_(.*?)_', args.model_path).group(1)
    meta_ratio = re.search(r'metaratio_(.*?)_', args.model_path).group(1)
    alpha = re.search(r'alpha_(.*?)_', args.model_path).group(1)
    fusion = re.search(r'fusion_(.*?)_', args.model_path).group(1)
    seed = re.search(r'seed_(.*?).pth', args.model_path).group(1)
else:
    modulation = 'Normal'
    inverse = False
    bs = args.batch_size
    meta_ratio = args.meta_ratio
    alpha = -1.
    fusion = 'identical'
    seed = args.seed

import logging

if args.train:
    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('{}/{}_inverse_{}_bs_{}_fusion_{}_metaratio_{}_seed_{}.txt'.format(args.ckpt_path, modulation, inverse, bs, fusion, meta_ratio, seed))
    file_handler.setLevel(logging.INFO)

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(args)


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0


    for step, (spec, image, label) in enumerate(dataloader):

        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        out_a, out_v, _ = model(spec.unsqueeze(1).float(), image.float())

        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss = loss_a + loss_v
        loss.backward()

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()

        num = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            if args.snr > 0:
                if args.snrModality == 'visual':
                    image = image + torch.randn(image.shape) * math.sqrt(torch.mean(torch.pow(image, 2)) / math.pow(10, args.snr / 10))
                elif args.snrModality == 'audio':
                    spec = spec + torch.randn(spec.shape) * math.sqrt(torch.mean(torch.pow(spec, 2)) / math.pow(10, args.snr / 10))

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            out_a, out_v, _ = model(spec.unsqueeze(1).float(), image.float())

            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    return sum(acc_a) / sum(num), sum(acc_v) / sum(num)


def main():

    setup_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    model = AVClassifier(args)

    model.apply(weight_init)
    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    if len(args.model_path) > 0:
        # first load trained model
        loaded_dict = torch.load(args.model_path)
        state_dict = loaded_dict['model']
        model.load_state_dict(state_dict, strict=False)
        print("load well-trained model!")
    else:
        print("train from scratch！")

    for name, param in model.named_parameters():
        if 'fusion' not in name:  # 假设fc是分类层的名称; 根据您的模型结构进行更改
            param.requires_grad = False

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'VGGSound':
        train_dataset = VGGSound(args, mode='train')
        test_dataset = VGGSound(args, mode='test')
    elif args.dataset == 'KineticSound':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    elif args.dataset == 'CREMAD':
        train_dataset = CramedDataset(args, mode='train')
        test_dataset = CramedDataset(args, mode='test')
    elif args.dataset == 'AVE':
        train_dataset = AVDataset(args, mode='train')
        test_dataset = AVDataset(args, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

    if args.train:

        best_acc_a = 0.0
        best_acc_v = 0.0

        best_models_a = []
        best_models_v = []

        for epoch in range(args.epochs):

            logger.info('Epoch: {}: '.format(epoch))
            batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                 train_dataloader, optimizer, scheduler)
            acc_a, acc_v = valid(args, model, device, test_dataloader)

            global fusion
            if acc_a > best_acc_a:
                best_acc_a = float(acc_a)

                model_name = '{}_inverse_{}_alpha_{}_' \
                             'bs_{}_fusion_{}_metaratio_{}_' \
                             'epoch_{}_acc_a_{}_seed_{}.pth'.format(modulation,
                                                          inverse,
                                                          alpha,
                                                          bs,
                                                          fusion,
                                                          meta_ratio,
                                                          epoch, acc_a, seed)

                saved_dict = {'saved_epoch': epoch,
                              'acc_a': acc_a,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                # logger.info("Audio Acc: {:.3f} ".format(acc_a))

                # 更新已保存的最佳模型列表
                best_models_a.append((acc_a, save_dir))
                best_models_a.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

                # 如果保存的模型超过1个，则删除准确率最低的模型
                while len(best_models_a) > 1:
                    _, oldest_model_path = best_models_a.pop()  # 获取准确率最低的模型
                    os.remove(oldest_model_path)  # 删除该模型文件

            if acc_v > best_acc_v:
                best_acc_v = float(acc_v)

                model_name = '{}_inverse_{}_alpha_{}_' \
                             'bs_{}_fusion_{}_metaratio_{}_' \
                             'epoch_{}_acc_v_{}_seed_{}.pth'.format(modulation,
                                                          inverse,
                                                          alpha,
                                                          bs,
                                                          fusion,
                                                          meta_ratio,
                                                          epoch, acc_v, seed)

                saved_dict = {'saved_epoch': epoch,
                              'acc_v': acc_v,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                # logger.info("Visual Acc: {:.3f} ".format(acc_v))

                # 更新已保存的最佳模型列表
                best_models_v.append((acc_v, save_dir))
                best_models_v.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

                # 如果保存的模型超过1个，则删除准确率最低的模型
                while len(best_models_v) > 1:
                    _, oldest_model_path = best_models_v.pop()  # 获取准确率最低的模型
                    os.remove(oldest_model_path)  # 删除该模型文件

            logger.info("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

    else:
        # first load trained model
        loaded_dict = torch.load(args.model_path)
        # epoch = loaded_dict['saved_epoch']
        # alpha = loaded_dict['alpha']
        # fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']
        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc_a, acc_v = valid(args, model, device, test_dataloader)
        print('accuracy_a: {}, accuracy_v: {}'.format(acc_a, acc_v))


if __name__ == "__main__":
    main()
