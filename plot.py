import os
import numpy as np
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter1d

import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.pyplot import MultipleLocator
from pylab import *
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_lib = sns.color_palette()

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16


def to_percent(temp, position):
    return '%2.f'%(temp) + '%'

way = 'e'  # way 是三种图
dataset = "CREMA-D"  # CREMA-D


def extract_txt(file, unimodal=False):
    # 用于存储所有找到的准确率值
    accuracies = []
    a_accuracies = []
    v_accuracies = []
    if not unimodal:
        # 打开文件并逐行读取
        with open(file, 'r') as f:
            for line in f:
                # 使用正则表达式查找匹配的字符串
                match = re.search(r'Loss: \d+\.\d+, Acc: (\d+\.\d+)', line)
                if match:
                    # 将找到的准确率值转换为浮点数并添加到列表中
                    accuracies.append((1.0 - float(match.group(1))) * 100.)
        return accuracies
    else:
        # 打开文件并逐行读取
        with open(file, 'r') as f:
            for line in f:
                # 使用正则表达式查找匹配的字符串
                match = re.search(r'Audio Acc: (\d+\.\d+)， Visual Acc: (\d+\.\d+)', line)
                if match:
                    # 将找到的准确率值转换为浮点数并添加到列表中
                    a_accuracies.append((float(match.group(1))) * 100.)
                    v_accuracies.append((float(match.group(2))) * 100.)
        return a_accuracies, v_accuracies


def extract_txt_robust(file, unimodal=False):
    # 用于存储所有找到的准确率值
    accuracies = []

    # 打开文件并逐行读取
    with open(file, 'r') as f:
        for line in f:
            # 使用正则表达式查找匹配的字符串
            if unimodal == False:
                match = re.search(r'Accuracy: (\d+\.\d+), accuracy_a', line)
            else:
                match = re.search(r'accuracy_a: (\d+\.\d+), accuracy_v: \d+\.\d+', line)
            if match:
                # 将找到的准确率值转换为浮点数并添加到列表中
                accuracies.append((float(match.group(1))) * 100.)
    return accuracies[::-1]

# --------box plot--------#
if way == 'a':
    # 设置seaborn样式
    sns.set(style="whitegrid")

    # 数据, seed 0 1 1234; meta ratio 0.0, 0.01, 0.005, 0.1, 0.3, 0.5, 1.0
    data = [[60.6, 61.7, 63.4],
            [66.9, 65.7, 65.9],
            [68.4, 66.4, 65.5],
            [67.5, 67.6, 65.7],
            [67.7, 68.4, 66.1],
            [63.0, 63.4, 64.5],
            [61.0, 61.0, 59.4],
            [62.9, 65.2, 71.5]]

    # 创建箱线图
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=np.transpose(data))

    # 设置箱线图颜色，并调整亮度
    colors = ['blue', 'green', 'red', 'yellow', 'pink', 'salmon', 'gray', 'cyan']
    for i, patch in enumerate(ax.patches):
        # 调整颜色亮度，factor > 1 为变亮，factor < 1 为变暗
        lighter_color = mcolors.lighten_color(colors[i % len(colors)], factor=.5)
        patch.set_facecolor(lighter_color)

    # 设置x轴和y轴标签
    ax.set_xticklabels(['N/A', '0.0', '0.01', '0.05', '0.1', '0.3', '0.5', '1.0'])
    ax.set_xlabel('Hyperparameter Setting', fontsize=14, labelpad=10)
    ax.set_ylabel('Model Performance', fontsize=14, labelpad=10)

    # 设置标题
    ax.set_title('Model Performance Distribution for Different Hyperparameter Settings', fontsize=16, pad=20)

    # 显示网格
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

    # 增加背景线条
    for i in np.arange(59, 73, 2):
        ax.axhline(i, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

    # 设置刻度标签的大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 显示图表
    plt.tight_layout()
    plt.show()


# ----------error curve----------
if way == 'b':
    fig, ax = plt.subplots(figsize=(12, 6))
    if dataset == 'CREMA-D':
        seed_list = [0, 1234, 1]
        legend_list = ['Without BMF', "With BMF"]
        load_root = "/home/hexiang/OGM-GE_CVPR2022/result/"
        type_list = ["Without BMF", "With BMF"]
        acc_epoch_list = ["Normal_inverse_False_bs_64_fusion_concat_metaratio_-1.0_seed_{}.txt",
                          "OGM_GE_inverse_True_bs_64_fusion_metamodal_metaratio_0.1_seed_{}.txt"]
        show_epoch = 0
        sigma = 0.5  # 平滑系数

        for i in range(2):
            acc_lists = []
            for seed in seed_list:
                file = os.path.join(load_root, acc_epoch_list[i].format(seed))
                acc_list = extract_txt(file)
                acc_lists.append(acc_list)
            acc_mean = np.max(np.array(acc_lists), axis=1)
            acc_std = np.max(np.array(acc_lists), axis=1).std(0)
            print("for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], acc_mean, acc_mean.mean(), acc_std))
            # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

            acc_lists_mean_total = np.array(acc_lists).mean(0)[show_epoch:]
            acc_lists_std_total = np.array(acc_lists).std(0)[show_epoch:]

            acc_lists_mean_total = gaussian_filter1d(acc_lists_mean_total, sigma=sigma)
            acc_lists_std_total = gaussian_filter1d(acc_lists_std_total, sigma=sigma)

            ax.plot(range(1, len(acc_lists_mean_total) + 1), acc_lists_mean_total, linewidth=2, label=legend_list[i])
            ax.fill_between(range(1, len(acc_lists_mean_total) + 1), (acc_lists_mean_total - 1 * acc_lists_std_total), (acc_lists_mean_total + 1 * acc_lists_std_total), alpha=.3)

        ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0, fontsize=16)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Test Accuracy (Test set)', fontsize=16)

        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.show()
        # plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
        # plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
        # plt.savefig('{}.pdf'.format(dataset), dpi=300)


# ----------error bar----------
if way == 'c':
    without_bmf = []
    with_bmf = []

    if dataset == 'CREMA-D':
        without_bmf = [60.6, 63.4, 61.7, 63.3, 65.7, 65.8]
        with_bmf = [67.7, 66.1, 68.4, 69.4, 67.9, 71.4]
    elif dataset == 'nmnist-shd':
        without_bmf = [60.6, 63.4, 61.7, 63.3, 65.7, 65.8]
        with_bmf = [67.7, 66.1, 68.4, 69.4, 67.9, 71.4]
    elif dataset == 'mnist-dvs':
        without_bmf = [60.6, 63.4, 61.7, 63.3, 65.7, 65.8]
        with_bmf = [67.7, 66.1, 68.4, 69.4, 67.9, 71.4]

    ix3 = pd.MultiIndex.from_arrays([['Accuracy', 'Accuracy', 'Accuracy', 'mAP', 'mAP', 'mAP'], ['Accuracy', 'Accuracy', 'Accuracy', 'mAP', 'mAP', 'mAP']], names=['0', '1'])
    df3 = pd.DataFrame({'Without BMF': without_bmf, 'With BMF': with_bmf}, index=ix3)
    print(df3)

    # 分组
    gp3 = df3.groupby(level=0)
    print(gp3)
    means = gp3.mean()
    print(means)
    errors = gp3.std()
    means.plot.bar(yerr=errors, rot=0, capsize=15, xlabel=dataset, ylabel="Accuracy test", ylim=[59, 72])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.show()


# ----------unimodal curve----------
if way == 'd':
    fig, ax = plt.subplots(figsize=(12, 6))
    if dataset == 'CREMA-D':
        seed_list = [0, 1234, 1]
        legend_list = ["Unimodal", "Without BMF", "With BMF"]
        load_root = "/home/hexiang/OGM-GE_CVPR2022/result_unimodal/"
        type_list = ["Unimodal", "Without BMF", "With BMF"]
        acc_epoch_list = ["Normal_inverse_False_bs_64_fusion_identical_metaratio_-1.0_seed_{}.txt",
                          "OGM_GE_inverse_False_bs_64_fusion_concat_metaratio_-1.0_seed_{}.txt",
                          "OGM_GE_inverse_True_bs_64_fusion_ogmge_metaratio_0.1_seed_{}.txt"]  # OGM_GE_inverse_False_bs_64_fusion_concat_metaratio_-1.0_seed_0.txt, Normal_inverse_False_bs_64_fusion_concat_metaratio_-1.0_seed_{}.txt
        show_epoch = 0
        sigma = 0.1  # 平滑系数

        for i in range(3):
            a_acc_lists = []
            v_acc_lists = []
            for seed in seed_list:
                file = os.path.join(load_root, acc_epoch_list[i].format(seed))
                a_acc_list, v_acc_list = extract_txt(file, unimodal=True)
                a_acc_lists.append(a_acc_list)
                v_acc_lists.append(v_acc_list)
            # # -------audio--------
            # a_acc_mean = np.max(np.array(a_acc_lists), axis=1)
            # a_acc_std = np.max(np.array(a_acc_lists), axis=1).std(0)
            # print("Audio: for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], a_acc_mean, a_acc_mean.mean(), a_acc_std))
            # # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))
            #
            # a_acc_lists_mean_total = np.array(a_acc_lists).mean(0)[show_epoch:]
            # a_acc_lists_std_total = np.array(a_acc_lists).std(0)[show_epoch:]
            #
            # a_acc_lists_mean_total = gaussian_filter1d(a_acc_lists_mean_total, sigma=sigma)
            # a_acc_lists_std_total = gaussian_filter1d(a_acc_lists_std_total, sigma=sigma)
            #
            # ax.plot(range(1, len(a_acc_lists_mean_total) + 1), a_acc_lists_mean_total, linewidth=2, label="Audio:"+legend_list[i])
            # ax.fill_between(range(1, len(a_acc_lists_mean_total) + 1), (a_acc_lists_mean_total - 1 * a_acc_lists_std_total), (a_acc_lists_mean_total + 1 * a_acc_lists_std_total), alpha=.3)
            # ax.grid(True, linestyle='--', alpha=0.5)


            # -------visual--------
            v_acc_mean = np.max(np.array(v_acc_lists), axis=1)
            v_acc_std = np.max(np.array(v_acc_lists), axis=1).std(0)
            print("Visual: for {}, acc max:{}, acc max mean:{} acc var:{}".format(legend_list[i], v_acc_mean, v_acc_mean.mean(), v_acc_std))
            # print("acc list:{}".format(np.max(np.array(acc_lists), axis=1)))

            v_acc_lists_mean_total = np.array(v_acc_lists).mean(0)[show_epoch:]
            v_acc_lists_std_total = np.array(v_acc_lists).std(0)[show_epoch:]

            v_acc_lists_mean_total = gaussian_filter1d(v_acc_lists_mean_total, sigma=sigma)
            v_acc_lists_std_total = gaussian_filter1d(v_acc_lists_std_total, sigma=sigma)

            ax.plot(range(1, len(v_acc_lists_mean_total) + 1), v_acc_lists_mean_total, linewidth=2, label="Visual:"+legend_list[i])
            ax.fill_between(range(1, len(v_acc_lists_mean_total) + 1), (v_acc_lists_mean_total - 1 * v_acc_lists_std_total), (v_acc_lists_mean_total + 1 * v_acc_lists_std_total), alpha=.3)
            ax.grid(True, linestyle='--', alpha=0.5)


        ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0, fontsize=18)
        plt.xlabel('Epochs', fontsize=18)
        plt.ylabel('Accuracy (%)', fontsize=18)
        plt.title('CREMA-D-ANN', fontsize=18)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
        plt.show()


# --------robust curve--------
if way == 'e':
    fig, ax = plt.subplots(figsize=(12, 6))
    if dataset == 'CREMA-D':
        seed_list = [0, 1234, 1]
        legend_list = ['Without BMF', "With BMF"]
        load_root = "/home/hexiang/OGM-GE_CVPR2022/"
        type_list = ["Without BMF", "With BMF"]
        acc_epoch_list = ["output_audio.txt",
                          "output_ours_audio.txt"]
        unimodal_list = [True, False]
        show_epoch = 0
        sigma = 0.1  # 平滑系数

        for i in range(2):
            a_acc_lists = []
            v_acc_lists = []
            # for seed in seed_list:
            file = os.path.join(load_root, acc_epoch_list[i])
            acc_list = extract_txt_robust(file, unimodal_list[i])
            ax.plot(range(1, len(acc_list) + 1), acc_list, linewidth=2, label=type_list[i])

        ax.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0, fontsize=18)
        plt.show()


# --------audio-visual event localization--------
task = 'classification'  # [classification, localization]
if way == 'f':
    import matplotlib.pyplot as plt

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color1 = colors[0]  # 第一种颜色
    color2 = colors[1]  # 第二种颜色

    # 数据
    if task == 'classification':
        methods = ['Concat Method', 'Concat w/ OGM\_GE', 'Concat w/ BMF (Ours)']
        values = [60.6, 64.9, 67.7]
    elif task == 'localization':
        methods = ['PSP Method', 'PSP w/ OGM\_GE', 'PSP w/ BMF (Ours)']
        values = [76.2, 76.9, 77.4]

    # 创建柱状图
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, values, color=[color1, color1, color2])  # 使用灰色表示非Brain-Inspired方法

    # 添加标题和标签
    plt.title('Comparison of Methods')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Methods')

    # 仅为 "Our Method" 添加图注，使用与柱状图相同的颜色
    plt.legend([bars[2]], ['Brain-Inspired Method'])

    # 显示数值
    for i in range(len(values)):
        plt.text(i, values[i] + 0.1, f'{values[i]}%', ha='center')

    # 添加网格
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 限制纵坐标轴范围
    if task == 'classification':
        plt.ylim(58, 70)
    else:
        plt.ylim(75, 78)

    # 显示图形
    plt.show()