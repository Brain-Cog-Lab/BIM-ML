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
import math

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='Normal', type=str,
                        choices=['Normal', 'OGM_GE'])

    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['concat', 'metamodal'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--audio_path', default='/mnt/home/hexiang/datasets/CREMA-D/AudioWAV/', type=str)
    parser.add_argument('--visual_path', default='/mnt/home/hexiang/datasets/CREMA-D/', type=str)

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

    parser.add_argument('--use_tensorboard', action='store_true', help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')

    parser.add_argument('--inverse', action='store_true', help='inverse effectiveness')
    parser.add_argument('--meta_ratio', default=-1., type=float, help='meta ratio')
    # snr value
    parser.add_argument('--snr', default=0, type=float,
                        help='random noise amplitude controled by snr, 0 means no noise')
    parser.add_argument('--snrModality', type=str, help='which Modality')

    # rho value
    parser.add_argument('--rho', default=0., type=float,
                        help='rho value')
    parser.add_argument('--inverse-epoch', default=0, type=int)
    return parser.parse_args()

args = get_arguments()

if args.train:
    import logging

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('{}/{}_inverse_{}_bs_{}_fusion_{}_metaratio_{}_rho_{}_seed_{}.txt'.format(args.ckpt_path, args.modulation, args.inverse, args.batch_size, args.fusion_method, args.meta_ratio, args.rho, args.seed))
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
    log_name = '{}_inverse_{}_bs_{}_fusion_{}_metaratio_{}_rho_{}_seed_{}'.format(args.modulation, args.inverse, args.batch_size, args.fusion_method, args.meta_ratio, args.rho, args.seed)
    writer = SummaryWriter(os.path.join(writer_path, log_name))


def determine_states(tensor1, tensor2):
    """
    对两个Tensor进行元素相乘，并根据结果确定三种状态。

    :param tensor1: 第一个Tensor
    :param tensor2: 第二个Tensor
    :return: 包含状态标识的Tensor
    """
    # 确保两个Tensor具有相同的形状
    assert tensor1.shape == tensor2.shape, "Tensor的形状必须相同"

    # 状态0: 两个元素都是0
    state0 = (tensor1 == 0) & (tensor2 == 0)

    # 状态1: 一个元素是0，另一个是1
    state1 = (tensor1 != tensor2)

    # 状态2: 两个元素都是1
    state2 = (tensor1 == 1) & (tensor2 == 1)

    # 综合状态
    states = state0 * (1) + state1 * (0) + state2 * (-1)

    return states

def compute_inverse_loss(output_a, output_v, out, label, criterion, rho):
    """
        output_a: shape [batch, cls], i.e., [64, 6]
        output_v: shape [batch, cls], i.e., [64, 6]
        out: shape [batch, cls], i.e., [64, 6]
        rho: inverse coefficient
    """
    # 对简单的样本, 好区分, 单个模态信息足够, 融合幅度小一点 (给更小的loss); 对困难的样本, 难区分, 单个模态信息不足, 融合幅度大一些 (给更大的loss).
    # 上述这种做法不同于之前的, 得到了模态融合的输出才进行调整, 而是一种直接的原则.
    N = out.shape[1]

    # 获取每个样本最大概率的类别, 找出预测类别与真实标签匹配的样本, 匹配的mask是1.0, 才能进行正常操作
    predicted_classes = torch.argmax(out, dim=1)
    mask = (predicted_classes == label).float()

    # 对每个样本，找到最大和第二大的概率值, 判断难易, 大于阈值后的地方为1，区分性高, 代表简单
    top_two_probs, _ = torch.topk(output_a, 2, dim=1)
    diff_a = top_two_probs[:, 0] - top_two_probs[:, 1]
    threshold = 0.5 * 1. / (N - 1) * math.ceil(0.5 * (N - 1))
    inverse_a = (diff_a > threshold).float()


    # 对每个样本，找到最大和第二大的概率值, 判断难易, 大于阈值后的地方为1，区分性高, 代表简单
    top_two_probs, _ = torch.topk(output_v, 2, dim=1)
    diff_v = top_two_probs[:, 0] - top_two_probs[:, 1]
    threshold = 0.5 * 1. / (N - 1) * math.ceil(0.5 * (N - 1))
    inverse_v = (diff_v > threshold).float()

    inverse_tensor = determine_states(inverse_a, inverse_v)  # 0为困难, 1为难易各半, 2为简单

    out = inverse_tensor.unsqueeze(1) * out * rho
    out = out * mask.unsqueeze(1)  # 只有正确分类的才进行mask
    cls_loss = criterion(out, label)

    return cls_loss

def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    model.train()
    logger.info("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0


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
        if args.meta_ratio >= 0.0:
            if args.modulation == 'OGM_GE':
                output_a, output_v, disc_pred_a, disc_pred_v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                disc_pred_a, disc_pred_v, out = model(spec.unsqueeze(1).float(), image.float())
        else:
            if args.modulation == 'OGM_GE' and args.fusion_method == 'metamodal':
                output_a, output_v, disc_pred_a, disc_pred_v, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                a, v, out = model(spec.unsqueeze(1).float(), image.float())
                a = a.detach()
                v = v.detach()
                weight_size = model.module.fusion_module.fc_out.weight.size(1)
                output_v = (torch.mm(v, torch.transpose(model.module.fusion_module.fc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.module.fusion_module.fc_out.bias / 2)

                output_a = (torch.mm(a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.module.fusion_module.fc_out.bias / 2)


        inverse_loss = compute_inverse_loss(softmax(output_a), softmax(output_v), softmax(out), label, criterion, args.rho)

        cls_loss = criterion(out, label)

        if args.meta_ratio >= 0.0:
            loss_v = bce(disc_pred_v, visual_labels)
            loss_a = bce(disc_pred_a, audio_labels)
            loss = args.meta_ratio * (loss_a + loss_v) + cls_loss
        else:
            loss_v = criterion(output_v, label)
            loss_a = criterion(output_a, label)
            loss = cls_loss

        if epoch > args.inverse_epoch:  # 大于设定的才启用
            loss = loss + inverse_loss
        loss.backward()

        if args.modulation == 'Normal':
            # no modulation, regular optimization
            pass
        else:
            # Modulation starts here !
            score_v = sum([softmax(output_v)[i][label[i]] for i in range(output_v.size(0))])
            score_a = sum([softmax(output_a)[i][label[i]] for i in range(output_a.size(0))])
            score_av = sum([softmax(out)[i][label[i]] for i in range(out.size(0))])

            ratio_v = score_v / score_a
            ratio_a = 1 / ratio_v
            ratio_av = (score_a + score_v) / score_av

            """
            Below is the Eq.(10) in our CVPR paper:
                    1 - tanh(alpha * rho_t_u), if rho_t_u > 1
            k_t_u =
                    1,                         else
            coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

            if ratio_v > 1:
                coeff_v = 1 - tanh(args.alpha * relu(ratio_v))
                coeff_a = 1
            else:
                coeff_a = 1 - tanh(args.alpha * relu(ratio_a))
                coeff_v = 1
            coeff_av = 1 + tanh(torch.tensor(1.0)) - tanh(relu(ratio_av))  # a 和 v 越弱, av出来越强

            if args.use_tensorboard:
                iteration = epoch * len(dataloader) + step

                writer.add_scalars('score', {'a': score_a,
                                             'v': score_v,
                                             'av': score_av}, iteration)

                writer.add_scalars('Coefficient', {'a': coeff_a,
                                                   'v': coeff_v,
                                                   'av': coeff_av}, iteration)

            if args.modulation_starts <= epoch <= args.modulation_ends: # bug fixed
                for name, parms in model.named_parameters():
                    layer = str(name).split('.')[1]  # 因为是并行的所以有一个moudle前缀. 因此序列索引为1.

                    if 'audio' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_a + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                    if 'visual' in layer and len(parms.grad.size()) == 4:
                        if args.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_v + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

                    if 'fusion' in layer:
                        if args.inverse is True:
                            parms.grad *= coeff_av
            else:
                pass


        optimizer.step()

        _loss += loss.item()
        if args.meta_ratio == 0.0:
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()

    scheduler.step()

    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader)


def calculate_map(predictions, labels, n_classes):
    APs = []

    for class_id in range(n_classes):
        # 提取所有样本的当前类别的预测分数
        class_scores = predictions[:, class_id]
        true_class = (labels == class_id).float()

        # 按分数排序样本
        sorted_indices = torch.argsort(class_scores, descending=True)
        true_class = true_class[sorted_indices]

        # 计算每个样本的精度和召回
        tp = torch.cumsum(true_class, dim=0)
        fp = torch.cumsum(1 - true_class, dim=0)

        precision = tp / (tp + fp)
        recall = tp / true_class.sum()

        # 计算插值精度
        precision = torch.cat([torch.tensor([1]).cuda(), precision])
        recall = torch.cat([torch.tensor([0]).cuda(), recall])

        for i in range(precision.size(0) - 1, 0, -1):
            precision[i - 1] = torch.max(precision[i - 1], precision[i])

        # 计算AP
        AP = torch.sum((recall[1:] - recall[:-1]) * precision[1:])
        APs.append(AP)

    # 计算MAP
    MAP = torch.mean(torch.tensor(APs))
    return MAP.item()

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
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        predictions = []
        labels = []
        for step, (spec, image, label) in enumerate(dataloader):

            if args.snr > 0:
                if args.snrModality == 'visual':
                    image = image + torch.randn(image.shape) * math.sqrt(torch.mean(torch.pow(image, 2)) / math.pow(10, args.snr / 10))
                elif args.snrModality == 'audio':
                    spec = spec + torch.randn(spec.shape) * math.sqrt(torch.mean(torch.pow(spec, 2)) / math.pow(10, args.snr / 10))

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            if args.fusion_method == 'ogmge_metamodal':
                out_a, out_v, _, _, out = model(spec.unsqueeze(1).float(), image.float())
            else:
                out_a, out_v, out = model(spec.unsqueeze(1).float(), image.float())

            if args.fusion_method == 'concat':
                out_v = (torch.mm(out_v, torch.transpose(model.module.fusion_module.fc_out.weight[:, 512:], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)
                out_a = (torch.mm(out_a, torch.transpose(model.module.fusion_module.fc_out.weight[:, :512], 0, 1)) +
                         model.module.fusion_module.fc_out.bias / 2)

            predictions.append(out)
            labels.append(label)

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    predictions = torch.cat(predictions, dim=0)  # 假设你有一个保存所有预测值的列表
    labels = torch.cat(labels, dim=0)  # 假设你有一个保存所有真实标签的列表
    map_value = calculate_map(predictions, labels, n_classes)

    from sklearn.metrics import roc_auc_score
    labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=6).cpu().numpy()
    # 计算AUC-ROC
    roc_auc = roc_auc_score(labels_one_hot, predictions.cpu().numpy(), multi_class='ovr')
    print("roc_auc:{}".format(roc_auc))

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num), map_value


def main():

    setup_seed(args.seed)
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')

    if args.modulation == 'OGM_GE' and args.fusion_method == 'metamodal':
        args.fusion_method = 'ogmge_metamodal'

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

            batch_loss, batch_loss_a, batch_loss_v = train_epoch(args, epoch, model, device,
                                                                 train_dataloader, optimizer, scheduler)
            acc, acc_a, acc_v, _ = valid(args, model, device, test_dataloader)

            if args.use_tensorboard:
                # writer.add_scalars('Loss', {'Total Loss': batch_loss,
                #                             'Audio Loss': batch_loss_a,
                #                             'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            if acc > best_acc:
                best_acc = float(acc)

                if not os.path.exists(args.ckpt_path):
                    os.mkdir(args.ckpt_path)

                model_name = '{}_inverse_{}_alpha_{}_' \
                             'bs_{}_fusion_{}_metaratio_{}_' \
                             'epoch_{}_acc_{}_rho_{}_seed_{}.pth'.format(args.modulation,
                                                          args.inverse,
                                                          args.alpha,
                                                          args.batch_size,
                                                          args.fusion_method,
                                                          args.meta_ratio,
                                                          epoch, acc, args.rho, args.seed)

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
                logger.info("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

                # 更新已保存的最佳模型列表
                best_models.append((acc, save_dir))
                best_models.sort(key=lambda x: x[0], reverse=True)  # 按准确率降序排序

                # 如果保存的模型超过1个，则删除准确率最低的模型
                while len(best_models) > 1:
                    _, oldest_model_path = best_models.pop()  # 获取准确率最低的模型
                    os.remove(oldest_model_path)  # 删除该模型文件

            else:
                logger.info("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
                logger.info("Audio Acc: {:.3f}， Visual Acc: {:.3f} ".format(acc_a, acc_v))

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

        # assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
        # assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded!')

        acc, acc_a, acc_v, map = valid(args, model, device, test_dataloader)
        print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}, mAP: {}'.format(acc, acc_a, acc_v, map))


if __name__ == "__main__":
    main()