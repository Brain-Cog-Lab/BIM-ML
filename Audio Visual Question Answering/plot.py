import numpy as np
import matplotlib.pyplot as plt

method = "st-avqa"  # base, st-avqa

if method == "base":
    # -----------------------------
    # 1) 准备示例数据
    # -----------------------------
    mode = "audio-visual"  # audio, visual, audio-visual

    if mode == "audio":
        labels = ['Avg', 'Counting', 'Comparative']  # 3个维度标签
        baseline  = [71.60, 77.20, 62.06]
        ours   = [72.40, 77.40, 63.89]
    elif mode == "visual":
        labels = ['Avg', 'Counting', 'Location']  # 3个维度标签
        baseline  = [75.48, 74.15, 76.79]
        ours   = [76.02, 75.23, 76.79]
    elif mode == "audio-visual":
        labels = ['Avg', 'Existential', 'Location', 'Counting', 'Comparative', 'Temporal']  # 6个维度标签
        baseline  = [67.34, 81.71, 67.43, 62.57, 61.61, 62.99]
        ours   = [67.87, 82.11, 69.07, 59.55, 62.87, 64.93]
        # labels = ['Avg', 'Location', 'Counting', 'Comparative', 'Temporal']  # 6个维度标签
        # baseline  = [67.61, 68.45, 63.00, 60.45, 62.86]
        # ours   = [68.33, 69.47, 62.68, 62.51, 64.20]
    # 函数：闭合多边形
    def close_circle(values):
        return values + values[:1]

    ours  = close_circle(ours)
    baseline   = close_circle(baseline)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合角度

    # -----------------------------
    # 2) 创建一个极坐标子图（Radar Chart）
    # -----------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)  # 单张雷达图

    # (a) 设置雷达图起始角度：让第一个维度在正上方
    ax.set_theta_offset(np.pi / 2)
    # (b) 设置方向：逆时针
    ax.set_theta_direction(-1)

    # (c) 设置每条轴对应的角度标记
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14, pad=4)

    # (d) 如果你想限制半径范围，比如 0~1
    if mode == "audio":
        ax.set_ylim(62, 78)
        ax.set_rticks([65.0, 75])
    elif mode == "visual":
        ax.set_ylim(73, 77)
        ax.set_rticks([73, 75, 77])
    elif mode == "audio-visual":
        ax.set_ylim(59, 83)
        # ax.set_rticks([73, 75, 77])
    # -----------------------------
    # 3) 绘制多边形 & 填充
    # -----------------------------

    # 画 Baseline
    ax.plot(angles, baseline, color='C1', linestyle='--', linewidth=2, label='Vanilla')
    ax.fill(angles, baseline, color='C1', alpha=0.2)

    # 画 ours
    ax.plot(angles, ours, color='C0', linewidth=2, label='W/ IEMF')
    ax.fill(angles, ours, color='C0', alpha=0.2)

    # -----------------------------
    # 4) 设置图例、标题并显示
    # -----------------------------
    if mode == "audio":
        title = "Audio Question"
    elif mode == "visual":
        title = "Visual Question"
    else:
        title = "Audio-Visual Question"
    ax.set_title(title, y=1.08, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=14)
    plt.tight_layout()

else:
    # -----------------------------
    # 1) 准备示例数据
    # -----------------------------
    mode = "visual"  # audio, visual, audio-visual

    if mode == "audio":
        labels = ['Avg', 'Counting', 'Comparative']  # 3个维度标签
        baseline = [71.90, 77.59, 62.23]
        ours = [74.49, 79.84, 65.39]
    elif mode == "visual":
        labels = ['Avg', 'Counting', 'Location']  # 3个维度标签
        baseline = [74.74, 73.89, 75.57]
        ours = [75.65, 74.65, 76.63]
    elif mode == "audio-visual":
        labels = ['Avg', 'Existential', 'Location', 'Counting', 'Comparative', 'Temporal']  # 6个维度标签
        baseline = [67.61, 82.81, 68.45, 63.00, 60.45, 62.86]
        ours = [68.33, 82.11, 69.47, 62.68, 62.51, 64.20]
        # labels = ['Avg', 'Location', 'Counting', 'Comparative', 'Temporal']  # 6个维度标签
        # baseline  = [67.61, 68.45, 63.00, 60.45, 62.86]
        # ours   = [68.33, 69.47, 62.68, 62.51, 64.20]


    # 函数：闭合多边形
    def close_circle(values):
        return values + values[:1]


    ours = close_circle(ours)
    baseline = close_circle(baseline)

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合角度

    # -----------------------------
    # 2) 创建一个极坐标子图（Radar Chart）
    # -----------------------------
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)  # 单张雷达图

    # (a) 设置雷达图起始角度：让第一个维度在正上方
    ax.set_theta_offset(np.pi / 2)
    # (b) 设置方向：逆时针
    ax.set_theta_direction(-1)

    # (c) 设置每条轴对应的角度标记
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14, pad=4)

    # (d) 如果你想限制半径范围，比如 0~1
    if mode == "audio":
        ax.set_ylim(62, 80)
        ax.set_rticks([65.0, 75])
    elif mode == "visual":
        ax.set_ylim(73, 77)
        ax.set_rticks([73, 75, 77])
    elif mode == "audio-visual":
        ax.set_ylim(60, 83)
        # ax.set_rticks([73, 75, 77])
    # -----------------------------
    # 3) 绘制多边形 & 填充
    # -----------------------------

    # 画 Baseline
    ax.plot(angles, baseline, color='C1', linestyle='--', linewidth=2, label='Vanilla')
    ax.fill(angles, baseline, color='C1', alpha=0.2)

    # 画 ours
    ax.plot(angles, ours, color='C0', linewidth=2, label='w/ IEMF')
    ax.fill(angles, ours, color='C0', alpha=0.2)

    # -----------------------------
    # 4) 设置图例、标题并显示
    # -----------------------------
    if mode == "audio":
        title = "Audio Question"
    elif mode == "visual":
        title = "Visual Question"
    else:
        title = "Audio-Visual Question"
    ax.set_title(title, y=1.08, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=14)
    plt.tight_layout()

plt.savefig('./save_figs/{}_{}.svg'.format(method, mode), dpi=300)
plt.show()