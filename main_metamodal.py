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


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM', 'OGM_GE'])

    parser.add_argument('--fusion_method', default='metamodal', type=str,
                        choices=['concat', 'metamodal'])
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

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')

    parser.add_argument('--inverse', action='store_true', help='inverse effectiveness')

    parser.add_argument('--meta_ratio', default=0.1, type=float, help='meta ratio')
    return parser.parse_args()

args = get_arguments()

import logging

# 创建一个logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建一个handler，用于写入日志文件
file_handler = logging.FileHandler(os.path.join(args.ckpt_path, '{}_inverse_{}_bs_{}_fusion_{}_metaratio_{}_seed_{}.txt'.format(args.modulation, args.inverse, args.batch_size, args.fusion_method, args.meta_ratio, args.seed)))
file_handler.setLevel(logging.INFO)

# 创建一个handler，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 给logger添加handler
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(args)

if args.use_tensorboard:
    writer_path = os.path.join(args.tensorboard_path, args.dataset)
    if not os.path.exists(writer_path):
        os.mkdir(writer_path)
    log_name = '{}_inverse_{}_bs_{}_fusion_{}_metaratio_{}_seed_{}'.format(args.modulation, args.inverse, args.batch_size, args.fusion_method, args.meta_ratio, args.seed)
    writer = SummaryWriter(os.path.join(writer_path, log_name))


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCELoss()

    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0


    coeff_av_max = -1
    for step, (spec, image, label) in enumerate(dataloader):

        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)


        # 判别器的标签，音频为0，视觉为1
        audio_labels = torch.zeros(spec.shape[0], 1).to(device)
        visual_labels = torch.ones(image.shape[0], 1).to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        disc_pred_a, disc_pred_v, out = model(spec.unsqueeze(1).float(), image.float())

        cls_loss = criterion(out, label)
        if args.fusion_method != 'concat':
            loss_v = bce(disc_pred_v, visual_labels)
            loss_a = bce(disc_pred_a, audio_labels)
            loss = args.meta_ratio * (loss_a + loss_v) + cls_loss
        else:
            loss = cls_loss
        loss.backward()

        optimizer.step()

        _loss += loss.item()
        if args.fusion_method != 'concat':
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
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            _, _, out = model(spec.unsqueeze(1).float(), image.float())

            prediction = softmax(out)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                num[label[i]] += 1.0

                #pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num), _, _


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

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
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

        best_acc = 0.0

        best_models = []

        for epoch in range(args.epochs):

            logger.info('Epoch: {}: '.format(epoch))

            if args.use_tensorboard:

                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                # writer.add_scalars('Loss', {'Total Loss': batch_loss,
                #                             'Audio Loss': batch_loss_a,
                #                             'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                     train_dataloader, optimizer, scheduler)
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = '{}_inverse_{}_alpha_{}_' \
                             'bs_{}_fusion_{}_metaratio_{}_' \
                             'epoch_{}_acc_{}_seed_{}.pth'.format(args.modulation,
                                                          args.inverse,
                                                          args.alpha,
                                                          args.batch_size,
                                                          args.fusion_method,
                                                          args.meta_ratio,
                                                          epoch, acc, args.seed)

                saved_dict = {'saved_epoch': epoch,
                              'modulation': args.modulation,
                              'alpha': args.alpha,
                              'fusion': args.fusion_method,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.ckpt_path, model_name)

                torch.save(saved_dict, save_dir)
                logger.info('The best model has been saved at {}.'.format(save_dir))
                logger.info("Loss: {:.3f}, Acc: {:.3f}".format(batch_loss, acc))

                # 更新已保存的最佳模型列表
                best_models.append((acc, save_dir))
                best_models.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

                # 如果保存的模型超过1个，则删除准确率最低的模型
                while len(best_models) > 1:
                    _, oldest_model_path = best_models.pop()  # 获取准确率最低的模型
                    os.remove(oldest_model_path)  # 删除该模型文件

            else:
                logger.info("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))

    else:
        # first load trained model
        loaded_dict = torch.load(args.ckpt_path)
        # epoch = loaded_dict['saved_epoch']
        modulation = loaded_dict['modulation']
        # alpha = loaded_dict['alpha']
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']
        # optimizer_dict = loaded_dict['optimizer']
        # scheduler = loaded_dict['scheduler']

        assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        logger.info('Trained model loaded!')

        acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
        logger.info('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()
