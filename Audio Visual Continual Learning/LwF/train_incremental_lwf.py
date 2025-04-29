import os
import sys
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.dirname(current_dir)
# 添加到 path 中
sys.path.append(project_root)

from dataloader_lwf import IcreLoader
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
from model.audio_visual_model_incremental import IncreAudioVisualNet
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np
from datetime import datetime
import random
import h5py
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

os.environ["OMP_NUM_THREADS"] = "20"  # 设置OpenMP计算库的线程数
os.environ["MKL_NUM_THREADS"] = "20"  # 设置MKL-DNN CPU加速库的线程数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AVE', choices=['AVE', 'ksounds', 'VGGSound_100'])
parser.add_argument('--modality', type=str, default='visual', choices=['visual', 'audio', 'audio-visual'])
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--infer_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--max_epoches', type=int, default=500)
parser.add_argument('--num_classes', type=int, default=28)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=boolean_string, default=False)
parser.add_argument("--milestones", type=int, default=[500], nargs='+', help="")
parser.add_argument('--seed', type=int, default=2025)

parser.add_argument('--class_num_per_step', type=int, default=7)

# 新加的
parser.add_argument('--e_prompt', action='store_true', default=False)
parser.add_argument('--prompt_dim', type=int, default=768)

parser.add_argument('--transfer', action='store_true', default=False)
parser.add_argument('--warm', action='store_true', default=False)

# for inverse
parser.add_argument('--inverse', action='store_true', help='inverse effectiveness')
parser.add_argument('--inverse_starts', default=0, type=int, help='where modulation begins')
parser.add_argument('--inverse_ends', default=100, type=int, help='where modulation ends')

args = parser.parse_args()

ckpts_root = './save/{}/{}/use-inverse_{}-seed_{}/'.format(args.dataset, args.modality, args.inverse, args.seed)
figs_root = './save/{}/{}/use-inverse_{}-seed_{}/fig'.format(args.dataset, args.modality, args.inverse, args.seed)

if not os.path.exists(ckpts_root):
    os.makedirs(ckpts_root)
if not os.path.exists(figs_root):
    os.makedirs(figs_root)

def Prepare_logger(args):
    import logging
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)

    logfile = os.path.join(ckpts_root, 'train.log')

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = Prepare_logger(args)

def CE_loss(num_classes, logits, label):
    targets = F.one_hot(label, num_classes=num_classes)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

    return loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def adjust_learning_rate(args, optimizer, epoch):
    miles_list = np.array(args.milestones) - 1
    if epoch in miles_list:
        for i, param_group in enumerate(optimizer.param_groups):
            current_lr = param_group['lr']
            new_lr = current_lr * 0.1
            logger.info('Reduce lr from {} to {}'.format(current_lr, new_lr))
            param_group['lr'] = new_lr

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    if dataset.args.dataset == 'AVE':
        dataset.all_visual_pretrained_features = np.load(dataset.visual_pretrained_feature_path, allow_pickle=True).item()
    else:
        dataset.all_visual_pretrained_features = h5py.File(dataset.visual_pretrained_feature_path, 'r')
    dataset.all_audio_pretrained_features = np.load(dataset.audio_pretrained_feature_path, allow_pickle=True).item()


def train(args, step, train_data_set, val_data_set):
    T = 2

    train_loader = DataLoader(train_data_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True, shuffle=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False, persistent_workers=True, worker_init_fn=worker_init_fn)
    
    step_out_class_num = (step + 1) * args.class_num_per_step
    if step == 0:
        model = IncreAudioVisualNet(args, step_out_class_num)
    else:
        model = torch.load(os.path.join(ckpts_root, 'step_{}_best_{}_model.pkl'.format(step - 1, args.modality)))
        model.incremental_classifier(step_out_class_num)
        old_model = torch.load(os.path.join(ckpts_root, 'step_{}_best_{}_model.pkl'.format(step - 1, args.modality)))
        last_step_out_class_num = step * args.class_num_per_step

        # 新老模型需要固定前t个时刻的prompt pool
        if args.e_prompt:
            with torch.no_grad():
                # Transfer previous learned prompt params to the new prompt
                if args.transfer:
                    a_prompts = [old_model.a_prompt.prompt[i].data for i in range(step)]
                    v_prompts = [old_model.v_prompt.prompt[i].data for i in range(step)]
                    model.a_prompt.prompt[step].data = torch.stack(a_prompts).mean(dim=0)
                    model.v_prompt.prompt[step].data = torch.stack(v_prompts).mean(dim=0)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if step != 0:
            old_model = nn.DataParallel(old_model)
    
    model = model.to(device)
    if step != 0:
        old_model = old_model.to(device)
        old_model.eval()

    if args.e_prompt:
        # 为prompt设置不同的学习率
        a_prompt_params = list(model.a_prompt.parameters())
        v_prompt_params = list(model.v_prompt.parameters())
        a_att_prompt_params = list(model.a_prompt_attention.parameters())
        v_att_prompt_params = list(model.v_prompt_attention.parameters())
        prompt_params = a_prompt_params + v_prompt_params
        prompt_att_params = a_att_prompt_params + v_att_prompt_params

        excluded_params = set(prompt_params) | set(prompt_att_params)

        base_params = [p for p in model.parameters() if p not in excluded_params]

        opt = torch.optim.Adam([
            {'params': base_params, "lr":args.lr / 2.},
            {'params': prompt_params, "lr":args.lr * 2},
            {'params': prompt_att_params, "weight_decay": args.weight_decay * 2},
        ], lr=args.lr, weight_decay=args.weight_decay)
        # opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dataset == "VGGSound_100":
        if args.inverse:
            # 设置分组参数
            classifier_params = []
            other_params = []

            for name, param in model.named_parameters():
                if 'classifier' in name:  # 或其他你模型中 classifier 的标志
                    classifier_params.append(param)
                else:
                    other_params.append(param)

            # 创建带有初始学习率的优化器
            opt = torch.optim.Adam([
                {'params': classifier_params, 'lr': args.lr},  # 这个会被之后动态修改
                {'params': other_params, 'lr': args.lr}
            ], weight_decay=args.weight_decay)

            classifier_param_ids = set(id(p) for p in classifier_params)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

    train_loss_list = []
    val_acc_list = []
    best_val_res = 0.0

    softmax = nn.Softmax(dim=1)
    tanh = nn.Tanh()

    for epoch in range(args.max_epoches):
        train_loss = 0.0
        num_steps = 0
        model.train()

        iterator = tqdm(train_loader)
        
        for samples in iterator:
            data, labels = samples
            labels = labels.to(device)
            if args.modality == 'visual':
                visual = data
                visual = visual.to(device)
                out = model(visual=visual)
            elif args.modality == 'audio':
                audio = data
                audio = audio.to(device)
                out = model(audio=audio)
            else:
                visual = data[0]
                audio = data[1]
                visual = visual.to(device)
                audio = audio.to(device)
                if args.e_prompt:
                    task_ids = torch.div(labels, args.class_num_per_step, rounding_mode='trunc').unsqueeze(1)
                    out, a_sim, v_sim = model(visual=visual, audio=audio, task_id=task_ids)
                else:
                    output_a, output_v, out = model(visual=visual, audio=audio)

            if step == 0:
                loss = CE_loss(step_out_class_num, out, labels)

                if args.inverse_starts <= epoch <= args.inverse_ends and args.inverse:
                    loss_a = CE_loss(args.class_num_per_step, output_a, labels)
                    loss_v = CE_loss(args.class_num_per_step, output_v, labels)

                    loss_single_modal = loss_a + loss_v  # 这里需要更新单模态的分类头
                    loss = loss + loss_single_modal

                    score_v = sum([softmax(output_v)[i][labels[i]] for i in range(output_v.size(0))])
                    score_a = sum([softmax(output_a)[i][labels[i]] for i in range(output_a.size(0))])
                    score_av = sum([softmax(out)[i][labels[i]] for i in range(out.size(0))])

                    ratio_av = ((score_a + score_v) / 2) / score_av
                    coeff_av = 1 + tanh(1. - ratio_av)  # a 和 v 越弱, av出来越强

                if args.e_prompt:
                    loss = loss - a_sim - v_sim
            else:
                with torch.no_grad():
                    if args.modality == 'visual':
                        old_out = old_model(visual=visual).detach()
                    elif args.modality == 'audio':
                        old_out = old_model(audio=audio).detach()
                    else:
                        if args.e_prompt:
                            task_ids = torch.div(labels, args.class_num_per_step, rounding_mode='trunc').unsqueeze(1)

                            mask = (task_ids == step)  # 创建一个掩码，标记所有等于 step 的位置
                            random_values = torch.randint(0, step, (task_ids.size(0), 1)).to(device)  # 生成随机值

                            # 使用掩码和 where 函数进行条件替换
                            task_ids_for_kl = torch.where(mask, random_values, task_ids)

                        if args.e_prompt:
                            old_out, a_sim_old, v_sim_old = old_model(visual=visual, audio=audio, task_id=task_ids_for_kl)
                            old_out = old_out.detach()
                        else:
                            old_output_a, old_output_v, old_out = old_model(visual=visual, audio=audio)
                            old_output_a, old_output_v, old_out = old_output_a.detach(), old_output_v.detach(), old_out.detach()

                loss_KD = torch.zeros(step).to(device)

                for t in range(step):
                    start = t * args.class_num_per_step
                    end = (t + 1) * args.class_num_per_step

                    soft_target = F.softmax(old_out[:, start:end] / T, dim=1)
                    output_log = F.log_softmax(out[:, start:end] / T, dim=1)
                    loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)

                    if args.inverse_starts <= epoch <= args.inverse_ends and args.inverse:
                        soft_target = F.softmax(old_output_a[:, start:end] / T, dim=1)
                        output_log = F.log_softmax(output_a[:, start:end] / T, dim=1)
                        loss_KD[t] = loss_KD[t] + F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)

                        soft_target = F.softmax(old_output_v[:, start:end] / T, dim=1)
                        output_log = F.log_softmax(output_v[:, start:end] / T, dim=1)
                        loss_KD[t] = loss_KD[t] + F.kl_div(output_log, soft_target, reduction='batchmean') * (T ** 2)
                loss_KD = loss_KD.sum()
                
                out = out[:, last_step_out_class_num:]
                labels = labels % args.class_num_per_step
                # labels = labels.to(device)
                ce_loss_value = CE_loss(args.class_num_per_step, out, labels)
    
                loss = ce_loss_value

                if args.inverse_starts <= epoch <= args.inverse_ends and args.inverse:
                    output_a = output_a[:, last_step_out_class_num:]
                    output_v = output_v[:, last_step_out_class_num:]
                    loss_a = CE_loss(args.class_num_per_step, output_a, labels)
                    loss_v = CE_loss(args.class_num_per_step, output_v, labels)

                    loss_single_modal = loss_a + loss_v  # 这里需要更新单模态的分类头
                    loss = loss + loss_single_modal

                    score_v = sum([softmax(output_v)[i][labels[i]] for i in range(output_v.size(0))])
                    score_a = sum([softmax(output_a)[i][labels[i]] for i in range(output_a.size(0))])
                    score_av = sum([softmax(out)[i][labels[i]] for i in range(out.size(0))])

                    ratio_av = ((score_a + score_v) / 2) / score_av
                    coeff_av = 1 + tanh(1. - ratio_av)  # a 和 v 越弱, av出来越强

                set_epoch_start = 20 # 这个时间段
                set_epoch_end = 50 # 这个时间段

                kd_weight = min(1.0, max((epoch - set_epoch_start) / (set_epoch_end - set_epoch_start), 0.0))

                if args.e_prompt:
                    loss = loss - a_sim - v_sim
                    if args.warm:
                        loss += kd_weight * loss_KD
                    else:
                        loss += loss_KD
                else:
                    loss += loss_KD

            model.zero_grad()
            loss.backward()

            if args.dataset == "VGGSound_100":
                if args.inverse_starts <= epoch <= args.inverse_ends and args.inverse:
                    for param_group in opt.param_groups:
                        if any(id(p) in classifier_param_ids for p in param_group['params']):
                            param_group['lr'] = args.lr * coeff_av.item()
            else:
                clip_grad_norm_(model.parameters(), max_norm=10.0)

                if args.inverse_starts <= epoch <= args.inverse_ends:
                    for name, parms in model.named_parameters():
                        layer = str(name).split('.')[0]
                        if 'classifier' in layer:
                            if args.inverse is True:
                                parms.grad = parms.grad * coeff_av + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

            opt.step()
            train_loss += loss.item()
            num_steps += 1
        train_loss /= num_steps
        train_loss_list.append(train_loss)
        logger.info('Epoch:{} train_loss:{:.5f}'.format(epoch, train_loss))

        all_val_out_logits = torch.Tensor([])
        all_val_labels = torch.Tensor([])
        model.eval()
        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader):
                # val_labels = val_labels.to(device)
                if args.modality == 'visual':
                    val_visual = val_data
                    val_visual = val_visual.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(visual=val_visual)
                    else:
                        val_out_logits = model(visual=val_visual)
                elif args.modality == 'audio':
                    val_audio = val_data
                    val_audio = val_audio.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(audio=val_audio)
                    else:
                        val_out_logits = model(audio=val_audio)
                else:
                    val_visual = val_data[0]
                    val_audio = val_data[1]
                    val_visual = val_visual.to(device)
                    val_audio = val_audio.to(device)
                    if torch.cuda.device_count() > 1:
                        val_out_logits = model.module.forward(visual=val_visual, audio=val_audio, is_train=False)
                    else:
                        _, _, val_out_logits = model(visual=val_visual, audio=val_audio, is_train=False)
                val_out_logits = F.softmax(val_out_logits, dim=-1).detach().cpu()
                all_val_out_logits = torch.cat((all_val_out_logits, val_out_logits), dim=0)
                all_val_labels = torch.cat((all_val_labels, val_labels), dim=0)
        val_top1 = top_1_acc(all_val_out_logits, all_val_labels)
        val_acc_list.append(val_top1)
        logger.info('Epoch:{} val_res:{:.6f} '.format(epoch, val_top1))

        if val_top1 > best_val_res:
            best_val_res = val_top1
            logger.info('Saving best model at Epoch {}'.format(epoch))
            if torch.cuda.device_count() > 1:
                torch.save(model.module,
                           os.path.join(ckpts_root, 'step_{}_best_{}_model.pkl'.format(step, args.modality)))
            else:
                torch.save(model, os.path.join(ckpts_root, 'step_{}_best_{}_model.pkl'.format(step, args.modality)))

        plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig(os.path.join(figs_root, '{}_train_loss_step_{}.png'.format(args.modality, step)))
        plt.close()

        plt.figure()
        plt.plot(range(len(val_acc_list)), val_acc_list, label='val_acc')
        plt.legend()
        plt.savefig(os.path.join(figs_root, '{}_val_acc_step_{}.png'.format(args.modality, step)))
        plt.close()


        if args.lr_decay:
            adjust_learning_rate(args, opt, epoch)


def detailed_test(args, step, test_data_set, task_best_acc_list):
    logger.info("=====================================")
    logger.info("Start testing...")
    logger.info("=====================================")

    model = torch.load(os.path.join(ckpts_root, 'step_{}_best_{}_model.pkl'.format(step, args.modality)))

    model.to(device)

    test_loader = DataLoader(test_data_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False, shuffle=False)
    
    all_test_out_logits = torch.Tensor([])
    all_test_labels = torch.Tensor([])
    model.eval()
    with torch.no_grad():
        for test_data, test_labels in tqdm(test_loader):
            # test_labels = test_labels.to(device)
            if args.modality == 'visual':
                test_visual = test_data
                test_visual = test_visual.to(device)
                test_out_logits = model(visual=test_visual)
            elif args.modality == 'audio':
                test_audio = test_data
                test_audio = test_audio.to(device)
                test_out_logits = model(audio=test_audio)
            else:
                test_visual = test_data[0]
                test_audio = test_data[1]
                test_visual = test_visual.to(device)
                test_audio = test_audio.to(device)
                _, _, test_out_logits = model(visual=test_visual, audio=test_audio, is_train=False)
            test_out_logits = F.softmax(test_out_logits, dim=-1).detach().cpu()
            all_test_out_logits = torch.cat((all_test_out_logits, test_out_logits), dim=0)
            all_test_labels = torch.cat((all_test_labels, test_labels), dim=0)
    test_top1 = top_1_acc(all_test_out_logits, all_test_labels)
    logger.info("Incremental step {} Testing res: {:.6f}".format(step, test_top1))
    
    old_task_acc_list = []
    mean_old_task_acc = 0.

    for i in range(step+1):
        step_class_list = range(i*args.class_num_per_step, (i+1)*args.class_num_per_step)
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(all_test_labels.numpy() == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        i_labels = torch.Tensor(all_test_labels.numpy()[step_class_idxs])
        i_logits = torch.Tensor(all_test_out_logits.numpy()[step_class_idxs])
        i_acc = top_1_acc(i_logits, i_labels)
        mean_old_task_acc += i_acc
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        logger.info('forgetting: {:.6f}'.format(forgetting))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None
    task_best_acc_list.append(curren_step_acc)

    mean_old_task_acc = mean_old_task_acc / (step + 1)
    return mean_old_task_acc, forgetting



if __name__ == "__main__":
    logger.info(args)

    total_incremental_steps = args.num_classes // args.class_num_per_step

    setup_seed(args.seed)
    
    logger.info('Training start time: {}'.format(datetime.now()))

    train_set = IcreLoader(args=args, mode='train', modality=args.modality)
    val_set = IcreLoader(args=args, mode='val', modality=args.modality)
    test_set = IcreLoader(args=args, mode='test', modality=args.modality)


    task_best_acc_list = []

    step_forgetting_list = []

    step_accuracy_list = []

    for step in range(total_incremental_steps):
        train_set.set_incremental_step(step)
        val_set.set_incremental_step(step)
        test_set.set_incremental_step(step)

        logger.info('Incremental step: {}'.format(step))
        train(args, step, train_set, val_set)
        step_accuracy, step_forgetting = detailed_test(args, step, test_set, task_best_acc_list)
        step_accuracy_list.append(step_accuracy)
        if step_forgetting is not None:
            step_forgetting_list.append(step_forgetting)
    Mean_accuracy = np.mean(step_accuracy_list)
    logger.info('Average Accuracy: {:.6f}'.format(Mean_accuracy))
    Mean_forgetting = np.mean(step_forgetting_list)
    logger.info('Average Forgetting: {:.6f}'.format(Mean_forgetting))
    
    if args.dataset != 'AVE' and args.modality != 'audio':
        train_set.close_visual_features_h5()
        val_set.close_visual_features_h5()
        test_set.close_visual_features_h5()


