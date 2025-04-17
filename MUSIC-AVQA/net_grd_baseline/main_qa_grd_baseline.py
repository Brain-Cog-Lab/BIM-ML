import sys
sys.path.append("/mnt/home/hexiang/MUSIC-AVQA/")
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from net_grd_baseline.dataloader_qa_grd_baseline import *
from net_grd_baseline.nets_qa_grd_baseline import AVQA_Fusion_Net
import ast
import json
import numpy as np
import pdb

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Implementation of MUSIC Audio-Visual Question Answering')

parser.add_argument(
    "--audio_dir", type=str, default='/mnt/home/hexiang/MUSIC-AVQA/vggish', help="audio dir")
parser.add_argument(
    "--video_dir", type=str, default='/home/hexiang/MUSIC-AVQA/feats_hdf5/res18_14x14/', help="video dir")
parser.add_argument(
    "--st_dir", type=str, default='/mnt/home/hexiang/MUSIC-AVQA/data/feats/r2plus1d_18', help="video dir")

parser.add_argument(
    "--label_train", type=str, default="/mnt/home/hexiang/MUSIC-AVQA/data/json/avqa-train.json", help="train csv file")
parser.add_argument(
    "--label_val", type=str, default="/mnt/home/hexiang/MUSIC-AVQA/data/json/avqa-val.json", help="val csv file")
parser.add_argument(
    "--label_test", type=str, default="/mnt/home/hexiang/MUSIC-AVQA/data/json/avqa-test.json", help="test csv file")
parser.add_argument(
    '--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 16)')
parser.add_argument(
    '--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 60)')
parser.add_argument(
    '--lr', type=float, default=1e-2, metavar='LR', help='learning rate (default: 3e-4)')
parser.add_argument(
    "--model", type=str, default='AVQA_Fusion_Net', help="with model to use")
parser.add_argument(
    "--mode", type=str, default='train', help="with mode to use")
parser.add_argument(
    '--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument(
    '--log-interval', type=int, default=50, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument(
    "--model_save_dir", type=str, default='net_grd_baseline/models_grd_baseline/', help="model save dir")
parser.add_argument(
    "--checkpoint", type=str, default='net_grd_baseline', help="save model name")
parser.add_argument(
    '--gpu', type=str, default='0, 1', help='gpu device number')

# for inverse
parser.add_argument('--inverse', action='store_true', help='inverse effectiveness')
parser.add_argument('--inverse_starts', default=0, type=int, help='where modulation begins')
parser.add_argument('--inverse_ends', default=50, type=int, help='where modulation ends')

args = parser.parse_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args.device = device

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

ckpts_root = '/mnt/home/hexiang/MUSIC-AVQA/results_baseline/inverse_{}_withmodified'.format(args.inverse)
os.makedirs(ckpts_root, exist_ok=True)

def Prepare_logger():
    import logging
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)


    logfile = os.path.join(ckpts_root, '{}.log'.format(args.mode))

    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = Prepare_logger()

import warnings
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/MUSIC-AVQA baseline/'+TIMESTAMP)

logger.info("\n--------------- MUSIC-AVQA baseline --------------- \n")

def batch_organize(audio_data, posi_img_data, nega_img_data):

    # logger.info("audio data: ", audio_data.shape)
    (B, T, C) = audio_data.size()
    audio_data_batch=audio_data.view(B*T,C)
    batch_audio_data = torch.zeros(audio_data_batch.shape[0] * 2, audio_data_batch.shape[1])


    (B, T, C, H, W) = posi_img_data.size()
    posi_img_data_batch=posi_img_data.view(B*T,C,H,W)
    nega_img_data_batch=nega_img_data.view(B*T,C,H,W)


    batch_image_data = torch.zeros(posi_img_data_batch.shape[0] * 2, posi_img_data_batch.shape[1], posi_img_data_batch.shape[2],posi_img_data_batch.shape[3])
    batch_labels = torch.zeros(audio_data_batch.shape[0] * 2)
    for i in range(audio_data_batch.shape[0]):
        batch_audio_data[i * 2, :] = audio_data_batch[i, :]
        batch_audio_data[i * 2 + 1, :] = audio_data_batch[i, :]
        batch_image_data[i * 2, :] = posi_img_data_batch[i, :]
        batch_image_data[i * 2 + 1, :] = nega_img_data_batch[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return batch_audio_data, batch_image_data, batch_labels



def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    softmax = nn.Softmax(dim=1)
    tanh = nn.Tanh()

    for batch_idx, sample in enumerate(train_loader):
        audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

        optimizer.zero_grad()
        a_out_qa, v_out_qa, out_qa, out_match,match_label = model(audio, visual_posi,visual_nega, question)
        loss_qa = criterion(out_qa, target)
        loss=loss_qa

        if args.inverse:
            loss_a = criterion(a_out_qa, target)
            loss_v = criterion(v_out_qa, target)

            loss_single_modal = loss_a + loss_v  # 这里需要更新单模态的分类头
            loss = loss + loss_single_modal

            score_v = sum([softmax(a_out_qa)[i][target[i]] for i in range(a_out_qa.size(0))])
            score_a = sum([softmax(v_out_qa)[i][target[i]] for i in range(v_out_qa.size(0))])
            score_av = sum([softmax(out_qa)[i][target[i]] for i in range(out_qa.size(0))])

            ratio_av = ((score_a + score_v) / 2) / score_av
            coeff_av = 1 + tanh(1. - ratio_av)  # a 和 v 越弱, av出来越强

        writer.add_scalar('data/both',loss.item(), epoch * len(train_loader) + batch_idx)

        loss.backward()
        if args.inverse_starts <= epoch <= args.inverse_ends:
            for name, parms in model.named_parameters():
                layer = str(name).split('.')[1]
                if 'fc_fusion' in layer:
                    if args.inverse is True:
                        parms.grad = parms.grad * coeff_av + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader,epoch):
    model.eval()
    total_qa = 0
    total_match=0
    correct_qa = 0
    correct_match=0
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            _, _, preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)

            _, predicted = torch.max(preds_qa.data, 1)
            total_qa += preds_qa.size(0)
            correct_qa += (predicted == target).sum().item()


    logger.info('Accuracy qa: %.2f %%' % (100 * correct_qa / total_qa))
    writer.add_scalar('metric/acc_qa',100 * correct_qa / total_qa, epoch)

    return 100 * correct_qa / total_qa


def test(model, val_loader):
    model.eval()
    total = 0
    correct = 0
    samples = json.load(open('/mnt/home/hexiang/MUSIC-AVQA/data/json/avqa-test.json', 'r'))
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []
    ans_list = ['two', 'cello', 'congas', 'zero', 'no', 'pipa', 'six', 'yes', 'one', 'four', 'three', 'seven', 'five', 'ukulele',
     'right', 'piano', 'left', 'accordion', 'clarinet', 'guzheng', 'more than ten', 'nine', 'indoor', 'saxophone',
     'drum', 'violin', 'middle', 'outdoor', 'bagpipe', 'bassoon', 'acoustic_guitar', 'banjo', 'electric_bass', 'ten',
     'eight', 'flute', 'simultaneously', 'trumpet', 'erhu', 'xylophone', 'tuba', 'suona']

    que_id=[]
    pred_results=[]
    
    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio,visual_posi,visual_nega, target, question, question_id = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda'), sample['question_id']

            _, _, preds_qa,preds_match,match_label = model(audio, visual_posi,visual_nega, question)
            preds=preds_qa
            _, predicted = torch.max(preds.data, 1)

            total += preds.size(0)
            correct += (predicted == target).sum().item()

            # save pred results
            pred_bool=predicted == target
            for index in range(len(pred_bool)):
                pred_results.append(pred_bool[index].cpu().item())
                que_id.append(question_id[index].item())

            x = samples[batch_idx]
            type =ast.literal_eval(x['type'])
            if type[0] == 'Audio':
                if type[1] == 'Counting':
                    A_count.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    A_cmp.append((predicted == target).sum().item())
            elif type[0] == 'Visual':
                if type[1] == 'Counting':
                    V_count.append((predicted == target).sum().item())
                elif type[1] == 'Location':
                    V_loc.append((predicted == target).sum().item())
            elif type[0] == 'Audio-Visual':
                if type[1] == 'Existential':
                    AV_ext.append((predicted == target).sum().item())
                    # AV_ext.append((predicted == target).sum().item())
                elif type[1] == 'Counting':
                    AV_count.append((predicted == target).sum().item())
                    if predicted != target:
                        print("not eq: {}".format(batch_idx))
                elif type[1] == 'Location':
                    AV_loc.append((predicted == target).sum().item())
                elif type[1] == 'Comparative':
                    AV_cmp.append((predicted == target).sum().item())
                elif type[1] == 'Temporal':
                    AV_temp.append((predicted == target).sum().item())

    # with open("pred_results/net_grd_baseline.txt", 'w') as f:
    #     for index in range(len(que_id)):
    #         f.write(str(que_id[index])+' '+str(pred_results[index]) + '\n')

    logger.info('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    logger.info('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    logger.info('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    logger.info('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    logger.info('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    logger.info('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    logger.info('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    logger.info('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    logger.info('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    logger.info('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    logger.info('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    logger.info('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    logger.info('Overall Accuracy: %.2f %%' % (
            100 * correct / total))

    return 100 * correct / total

def main():
    torch.manual_seed(args.seed)

    if args.model == 'AVQA_Fusion_Net':
        model = AVQA_Fusion_Net()
        model = nn.DataParallel(model)
        model = model.to('cuda')
    else:
        raise ('not recognized')

    if args.mode == 'train':
        train_dataset = AVQA_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_dataset = AVQA_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


        # ===================================== load pretrained model ===============================================
        # None
        # ===================================== load pretrained model ===============================================

        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader,epoch)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), os.path.join(ckpts_root, args.checkpoint + ".pt"))

    else:
        test_dataset = AVQA_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]))
        logger.info(test_dataset.__len__())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        model.load_state_dict(torch.load(os.path.join(ckpts_root, args.checkpoint + ".pt")))
        test(model, test_loader)


if __name__ == '__main__':
    main()