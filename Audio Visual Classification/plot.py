import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator
import matplotlib

# # -----------------------1. plot accuracy -> fig 2(a), fig 2(b)-----------------------
# # Example data (fabricated for illustration)
# model_type = "SNN"  # ANN, SNN
# dataset = "CREMAD"  # CREMAD, KineticSound, UrbanSound8K
# methods = ['Concat', 'MSLR', 'OGG_GE', 'LFM']
# if model_type == "ANN":
#     if dataset == "CREMAD":
#         base_scores = [62.63, 64.11, 68.68, 64.11]
#         winv_scores = [63.44, 65.59, 71.10, 63.98]
#     elif dataset == "KineticSound":
#         base_scores = [51.58, 51.89, 57.63, 55.28]
#         winv_scores = [56.17, 55.86, 64.61, 63.15]
#     elif dataset == "UrbanSound8K":
#         base_scores = [97.90, 97.79, 97.60, 98.05]
#         winv_scores = [97.86, 97.98, 99.24, 98.63]
# else:
#     if dataset == "CREMAD":
#         base_scores = [63.04, 63.70, 69.49, 63.58]
#         winv_scores = [63.58, 64.38, 67.61, 64.78]
#     elif dataset == "KineticSound":
#         base_scores = [52.85, 53.19, 57.52, 54.63]
#         winv_scores = [55.47, 56.63, 61.41, 63.53]
#     elif dataset == "UrbanSound8K":
#         base_scores = [98.02, 97.98, 97.67, 98.13]
#         winv_scores = [97.79, 98.05, 99.35, 98.63]
#
# x = np.arange(len(methods))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots(figsize=(8, 6))
#
# bars_base = ax.bar(x - width/2, base_scores, width, label='Vanilla', color='#769AC9')
# bars_winv = ax.bar(x + width/2, winv_scores, width, label='w/ IEMF', color='#D5C5A9')
#
# # Add text labels above each bar
# for bar in bars_base + bars_winv:
#     height = bar.get_height()
#     ax.annotate(f'{height:.2f}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),  # offset in points
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=15)
#
# # Customise the axes and title
# ax.set_ylabel('Classification Accuracy (%)', fontsize=18)
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
#
# if dataset == "CREMAD":
#     ax.set_ylim(60, 75)
# elif dataset == "KineticSound":
#     ax.set_ylim(50, 70)
# elif dataset == "UrbanSound8K":
#     ax.set_ylim(97, 100)
#
# ax.tick_params(labelsize=18)
# ax.legend(fontsize=16)
#
# if dataset == "UrbanSound8K":
#     ax.yaxis.set_major_locator(MultipleLocator(1.0))
# else:
#     ax.yaxis.set_major_locator(MultipleLocator(4))
#
# # 去掉上边框和右边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.tight_layout()
# # plt.show()
# plt.savefig('./save_figs/{}_{}.svg'.format(model_type, dataset), dpi=300)
# plt.show()
#
#
# -----------------------2. plot joint comparison fig 2(c2)-----------------------

# # Example data (fabricated for illustration)
# methods = ['Concat', 'MSLR', 'OGG_GE', 'LFM']
# base_scores = [51.58, 51.89, 57.63, 55.28]
# joint_scores = [54.86, 55.51, 60.87, 61.10]
# winv_scores = [56.17, 55.86, 64.61, 63.15]  # ← 你新增的数据
# x = np.arange(len(methods))
# width = 0.265  # 每个柱子的宽度
#
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # 画三组柱子
# bars_base = ax.bar(x - width, base_scores, width, label='Vanilla', color='#769AC9')
# bars_winv = ax.bar(x, joint_scores, width, label='w/ Joint Learning', color='#e2f0d9')
# bars_joint = ax.bar(x + width, winv_scores, width, label='w/ IEMF', color='#D5C5A9')  # 可换色
#
# # 添加柱子上的数值
# for bar in bars_base + bars_winv + bars_joint:
#     height = bar.get_height()
#     ax.annotate(f'{height:.2f}',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=12)
#
# # 设置坐标轴等样式
# ax.set_ylabel('Classification Accuracy (%)', fontsize=18)
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.set_ylim(50, 69)
# ax.tick_params(labelsize=14)
# ax.legend(fontsize=14)
#
# # 去掉多余边框
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# fig.tight_layout()
# plt.savefig('./save_figs/joint.svg', dpi=300)
# plt.show()

# # ---------------------- 3. plot hyperparameter scatter fig 2(c1)------------------#
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 模拟数据
# inverse_coef = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])  # inverse coefficient
# accuracy = np.array([60.41, 60.25, 64.61, 63.34, 62.18, 59.48])  # Accuracy
#
# # 设置颜色映射范围
# norm = plt.Normalize(vmin=56., vmax=65.)
#
# # 创建散点图
# fig, ax = plt.subplots(figsize=(12, 6))
# scatter = ax.scatter(
#     inverse_coef, accuracy,
#     c=accuracy, cmap='turbo',
#     norm=norm,
#     marker='h', s=200,
#     edgecolors='k', alpha=0.75
# )
#
# # 添加网格
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
#
# # 设置颜色条
# cbar = plt.colorbar(scatter, ax=ax)
#
# # 在 colorbar 上加一条横线标记 baseline
# baseline_accuracy = 57.63
# cbar.ax.axhline(y=baseline_accuracy, color='black', linewidth=1, linestyle='--')
# cbar.ax.text(1.1, norm(baseline_accuracy), 'w/o IEMF', va='center', ha='left', color='red', fontsize=10, transform=cbar.ax.transAxes)
#
# # cbar.ax.axhline(y=np.min(accuracy), color='black', linewidth=1, linestyle='--')
# # cbar.ax.text(1.1, norm(np.min(accuracy)), 'ours', va='center', ha='left', color='red', fontsize=10, transform=cbar.ax.transAxes)
#
# ax.tick_params(labelsize=18)
# # 设置标签
# ax.set_xlabel('Inverse Gain Coefficient', fontsize=18)
# ax.set_ylabel('Classification Accuracy (%)', fontsize=18)
#
# plt.tight_layout()
# plt.savefig('./save_figs/hyperparameter.svg', dpi=300)
# plt.show()


# # -------------- 4. unimodality finetuning fig 2(d)-------------#
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
color_lib = sns.color_palette()

def extract_csv(file):
    column = [0, 5]
    data = np.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, usecols=column)
    return data[:, 0], data[:, 1]

def to_percent(temp, position):
    return '%2.f'%(temp) + '%'

legend_list = ['Audio Unimodal', "Unimodal branch (MM) w/o IEMF", "Unimodal branch (MM) w/ IEMF"]

modality = "audio"
dataset = "KineticSound"
inverse = False
baseline_root = "/mnt/home/hexiang/Brain-inspired mechanisms/exp_results/"
finetuning_root = "/mnt/home/hexiang/Brain-inspired mechanisms/exp_finetuning/"

fig, ax = plt.subplots(figsize=(14, 6))  # (12, 6)
axins = ax.inset_axes([0.42, 0.35, 0.4, 0.25])  # [left, bottom, width, height]
for i in range(3):
    if i == 2:
        inverse = True
    suffix = "AVresnet18-{}-{}-Normal-inverse_{}-psai_1.0-fusion_concat-seed_2025-ReLUNode-1/summary.csv".format(dataset, modality, inverse)

    if i == 0:
        file = os.path.join(baseline_root, suffix)
    else:
        file = os.path.join(finetuning_root, suffix)
    epoch_list, acc_list = extract_csv(file)
    epoch_list, acc_list = np.array(epoch_list), np.array(acc_list)
    print("for {}, acc max:{}".format(legend_list[i], acc_list.max()))
    ax.plot(range(1, len(epoch_list) + 1), acc_list, linewidth=2, label=legend_list[i])
    axins.plot(acc_list)

axins.set_xlim(70, 100)
if dataset == "KineticSound":
    if modality == "audio":
        axins.set_ylim(35, 41)
    else:
        axins.set_ylim(55, 60)
elif dataset == "CREMAD":
    if modality == "audio":
        axins.set_ylim(35, 41)
    else:
        axins.set_ylim(50, 63)

# 添加连接线标注
ax.indicate_inset_zoom(axins)

ax.legend(bbox_to_anchor=(1, 0), loc=4, fontsize=18)
plt.xlabel('Training Epochs', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
# plt.show()
plt.grid(which='major', axis='x', ls=':', lw=0.8, color='c', alpha=0.5)
plt.grid(which='major', axis='y', ls=':', lw=0.8, color='c', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tight_layout()
if modality == "audio":
    ax.yaxis.set_major_locator(MultipleLocator(5))  # 设置y轴的主要刻度间隔
else:
    ax.yaxis.set_major_locator(MultipleLocator(15))  # 设置y轴的主要刻度间隔
plt.savefig('./save_figs/finetuning.svg', dpi=300)
plt.show()
# sys.exit()

# # # ----- 6. inverse coefficient vary fig 2(e)--------
#
# import os, numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set_context("notebook", font_scale=1.1,
#                 rc={"lines.linewidth": 2.2})
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# csv_file = ("/mnt/home/hexiang/Brain-inspired mechanisms/results0423/"
#             "AVresnet18-KineticSound-audio-visual-OGM_GE-inverse_True-"
#             "coef_1.0-psai_1.0-fusion_concat-seed_2025-ReLUNode-1/summary.csv")
#
# def read_two_cols(path, col_epoch, col_value):
#     data = np.loadtxt(path, delimiter=",", skiprows=1,
#                       usecols=[col_epoch, col_value])
#     return data[:, 0], data[:, 1]
#
# # --- 左轴：Test-accuracy（第 5 列） ---
# epoch, acc = read_two_cols(csv_file, 0, 5)
#
# # --- 右轴：coeff（只取第 10 列） ---
# _, coeff = read_two_cols(csv_file, 0, 9)        # ★ 这里改成 10 ★
#
# fig, ax_acc = plt.subplots(figsize=(14, 6))      # 左轴
# l1, = ax_acc.plot(epoch, acc, color="tab:blue", lw=2.5,
#                   label="Test accuracy")
# ax_acc.set_xlabel("Training Epochs", fontsize=20)
# ax_acc.set_ylabel("Accuracy (%)", fontsize=20)
# ax_acc.tick_params(labelsize=18)
# ax_acc.grid(ls=":", lw=0.8, color="c", alpha=0.5)
#
# # --- 右轴：coeff ---
# ax_coef = ax_acc.twinx()
# l2, = ax_coef.plot(epoch, coeff, color="tab:orange", ls='--', lw=2.5,
#                    label="IEMF coefficient")
# ax_coef.set_ylabel("IEMF Coefficient",
#                    fontsize=20)
# ax_coef.tick_params(axis='y')
#
# ax_coef.tick_params(labelsize=18)
# # 合并图例
# ax_acc.legend([l1, l2], ["Test accuracy", "IEMF coefficient"],
#               loc="upper right", fontsize=18)
#
# plt.tight_layout()
# plt.savefig('./save_figs/coeff.svg', dpi=300)
# plt.show()
