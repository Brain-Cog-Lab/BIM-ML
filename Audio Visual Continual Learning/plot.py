import os

import matplotlib.pyplot as plt
import numpy as np

method = "LwF"  # LwF, SSIL, ours
dataset = "VGGSound_100"  # AVE, ksounds, VGGSound_100
inverse = False
seed = [0, 2025, 3917]
data_root = "/mnt/home/hexiang/AV-CIL_ICCV2023/{}/save/{}/audio-visual".format(method, dataset)

# 解析日志数据
def parse_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        accuracy = float(lines[-2][-9:-1]) * 100.
        forgetting = float(lines[-1][-9:-1]) * 100.
    return accuracy, forgetting


baseline_acc_list = []
baseline_forgetting_list = []
ours_acc_list = []
ours_forgetting_list = []
for s in seed:
    suffix = "use-inverse_{}-seed_{}/train.log".format(False, s)
    accuracy, forgetting = parse_log(os.path.join(data_root, suffix))
    baseline_acc_list.append(accuracy), baseline_forgetting_list.append(forgetting * -1.)

    suffix = "use-inverse_{}-seed_{}/train.log".format(True, s)
    accuracy, forgetting = parse_log(os.path.join(data_root, suffix))
    ours_acc_list.append(accuracy), ours_forgetting_list.append(forgetting * -1.)

# ---------------------------------------------1. acc --------------------------------------------
# 模拟实验数据
labels = [method, 'With Inverse']
data = {
    method: baseline_acc_list,
    'With Inverse': ours_acc_list
}
print(baseline_acc_list)
print(ours_acc_list)
# 计算每组的均值和标准差
means = [np.mean(data[label]) for label in labels]
print(means)
std_devs = [np.std(data[label]) for label in labels]

# 设置颜色
colors = ['#769AC9', '#D5C5A9']  # D5C5A9, EFECE6

# 创建图形
fig, ax = plt.subplots(figsize=(6, 8))

# 绘制每组数据的点
for i, label in enumerate(labels):
    ax.scatter([i] * len(data[label]), data[label], edgecolor='black', color='white', s=10, zorder=5)

# 绘制每组数据的柱状图
bars = ax.bar(labels, means, yerr=std_devs, width=0.5, color=colors, capsize=15, edgecolor='black')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + std_devs[i] + 0.15, f'{height:.2f}', ha='center', va='bottom', fontsize=18)

ax.set_ylabel('Average accuracy (%)', fontsize=18)
ax.tick_params(labelsize=18)


if method == "LwF" and dataset == "AVE":
    ax.set_ylim(50, 58)
elif method == "SSIL" and dataset == "AVE":
    ax.set_ylim(50, 60)
elif method == "ours" and dataset == "AVE":
    ax.set_ylim(55, 65)

if method == "LwF" and dataset == "ksounds":
    ax.set_ylim(55, 65)
elif method == "SSIL" and dataset == "ksounds":
    ax.set_ylim(60, 67)
elif method == "ours" and dataset == "ksounds":
    ax.set_ylim(69, 74)

if method == "LwF" and dataset == "VGGSound_100":
    ax.set_ylim(57, 62)
elif method == "SSIL" and dataset == "VGGSound_100":
    ax.set_ylim(65, 72)
elif method == "ours" and dataset == "VGGSound_100":
    ax.set_ylim(68, 72)
xtick_labels = ['AV-CIL' if method == 'ours' else method, 'With Inverse']
ax.set_xticks([0, 1])
ax.set_xticklabels(xtick_labels, fontsize=18)

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示图形
plt.tight_layout()
plt.savefig(os.path.join(data_root, "{}_accuracy_overall.svg".format(method)), format="svg", bbox_inches="tight", dpi=300)  # ← 添加这一行
plt.show()

# ---------------------------------------------2. forgetting --------------------------------------------

# 模拟实验数据
labels = [method, 'With Inverse']
data = {
    method: baseline_forgetting_list,
    'With Inverse': ours_forgetting_list
}
print(baseline_forgetting_list)
print(ours_forgetting_list)
# 计算每组的均值和标准差
means = [np.mean(data[label]) for label in labels]
print(means)
std_devs = [np.std(data[label]) for label in labels]

# 设置颜色
colors = ['#769AC9', '#D5C5A9']  #'#6C8C5A'

# 创建图形
fig, ax = plt.subplots(figsize=(6, 8))

# 绘制每组数据的点
for i, label in enumerate(labels):
    ax.scatter([i] * len(data[label]), data[label], edgecolor='black', color='white', s=10, zorder=5)

# 绘制每组数据的柱状图，并使柱状图向下（使用负值）
bars = ax.bar(labels, np.array(means), yerr=std_devs, width=0.5, color=colors, alpha=0.7, capsize=15, edgecolor='black')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height - std_devs[i] - 0.3, f'{height:.2f}', ha='center', va='top', fontsize=12)


# 绘制y=0的虚线
ax.axhline(0, color='black', linestyle='--')

# 设置标题和标签
# ax.set_title('R-CIFAR-100', fontsize=16, fontname='Helvetica', fontweight='bold')
ax.set_ylabel('Backward transfer (%)', fontsize=14, fontname='Helvetica')
ax.set_ylim(-35, 0)  # 让y轴从0开始向下延伸

# 去掉上边框和右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 显示图形
plt.tight_layout()
plt.savefig(os.path.join(data_root, "forgetting_overall.pdf"), format="pdf", bbox_inches="tight")  # ← 添加这一行
plt.show()