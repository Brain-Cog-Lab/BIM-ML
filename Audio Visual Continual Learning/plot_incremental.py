import re
import os

import matplotlib.pyplot as plt
import numpy as np

method = "LwF"  # LwF, SSIL, ours
dataset = "AVE"  # AVE, ksounds, VGGSound_100
inverse = False
seed = [0, 2025, 3917]
data_root = "/mnt/home/hexiang/AV-CIL_ICCV2023/{}/save/{}/audio-visual".format(method, dataset)

# 新函数：提取每步的测试结果
def parse_all_steps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        step_accs = []
        for line in lines:
            match = re.search(r'Incremental step \d+ Testing res: ([0-9.]+)', line)
            if match:
                step_accs.append(float(match.group(1)) * 100.)
    return step_accs

# 收集每个种子的 step-wise accuracy
baseline_steps = []
inverse_steps = []

for s in seed:
    suffix = "use-inverse_{}-seed_{}/train.log".format(False, s)
    baseline_steps.append(parse_all_steps(os.path.join(data_root, suffix)))

    suffix = "use-inverse_{}-seed_{}/train.log".format(True, s)
    inverse_steps.append(parse_all_steps(os.path.join(data_root, suffix)))

# 计算平均值和标准差（不同种子的结果）
baseline_array = np.array(baseline_steps)
inverse_array = np.array(inverse_steps)

baseline_avg = np.mean(baseline_array, axis=0)
baseline_std = np.std(baseline_array, axis=0)

inverse_avg = np.mean(inverse_array, axis=0)
inverse_std = np.std(inverse_array, axis=0)

# 画图
fig, ax = plt.subplots(figsize=(6, 8))
x = np.arange(1, len(baseline_avg) + 1)

method_tmp = method
if method == "ours":
    method_tmp = "AV-CIL"
# Baseline 折线 + 误差棒
ax.errorbar(x, baseline_avg, yerr=baseline_std, label=method_tmp, fmt='o-', color='#769AC9',
            ecolor='#769AC9', capsize=5, elinewidth=1.5, markeredgecolor='black')

# With Inverse 折线 + 误差棒
ax.errorbar(x, inverse_avg, yerr=inverse_std, label='With Inverse', fmt='s-', color='#D5C5A9',
            ecolor='#D5C5A9', capsize=5, elinewidth=1.5, markeredgecolor='black')

# 图形美化
ax.set_xlabel('Number of Tasks', fontsize=18)
ax.set_ylabel('Accuracy (%)', fontsize=18)
ax.tick_params(labelsize=18)

# ax.set_title(f'{method} on {dataset}', fontsize=16)
ax.set_xticks(x)
# ax.set_ylim(50, 100)
ax.legend(fontsize=18)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(data_root, "{}_task_accuracy_overall.svg".format(method)), format="svg", bbox_inches="tight", dpi=300)  # ← 添加这一行
plt.show()