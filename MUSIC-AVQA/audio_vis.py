import matplotlib.pyplot as plt
import librosa
import librosa.display
from moviepy.editor import *

# 1. 从视频中提取音频
id = "00002565"
video_path = '/root/{}.mp4'.format(id)
audio_path = '/root/extracted_audio.wav'

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path, codec='pcm_s16le')

# 2. 使用librosa加载音频数据
y, sr = librosa.load(audio_path, sr=None)


# 3. 保存特定时刻的音频波形为SVG文件
# 选择的时刻为第3秒、第30秒、第57秒、第59秒（从1开始数）

# # 函数：提取并保存指定时间的音频波形
# def save_waveform_at_time(y, sr, times, filename_prefix):
#     for time in times:
#         # 计算对应的样本索引
#         start_sample = int(time * sr)
#         end_sample = start_sample + sr  # 获取1秒的样本数据
#
#         # 确保不会超出音频的长度
#         if end_sample > len(y):
#             end_sample = len(y)
#
#         # 提取对应的音频段
#         y_segment = y[start_sample:end_sample]
#
#         # 绘制波形
#         plt.figure(figsize=(10, 4))
#         librosa.display.waveshow(y_segment, sr=sr)
#
#         # 隐藏标题和坐标轴
#         ax = plt.gca()
#         ax.set_title('')  # 去掉标题
#         ax.get_xaxis().set_visible(False)  # 隐藏X轴
#         ax.get_yaxis().set_visible(False)  # 隐藏Y轴
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#
#         # 保存为SVG文件
#         plt.tight_layout()
#         plt.savefig(f'/root/{filename_prefix}_at_{time}_seconds.svg')
#         plt.close()
#
#
# # 保存第3, 30, 57, 59秒的音频波形
# save_waveform_at_time(y, sr, [8, 15, 43, 60], 'audio_waveform_{}'.format(id))


# 绘制波形
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)

# 隐藏标题和坐标轴
ax = plt.gca()
ax.set_title('')  # 去掉标题
ax.get_xaxis().set_visible(False)  # 隐藏X轴
ax.get_yaxis().set_visible(False)  # 隐藏Y轴
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 保存为SVG文件
plt.tight_layout()
plt.savefig(f'/root/audio.svg')
plt.close()

# # --- plot 多个的 ---
# # 计算每个分段的样本数量
# segment_samples = len(y) // 10
#
# # 3. 使用matplotlib分别可视化每个部分的音频波形
# for i in range(10):
#     plt.figure(figsize=(10, 4))
#
#     segment_start = i * segment_samples
#     segment_end = (i + 1) * segment_samples
#     librosa.display.waveshow(y[segment_start:segment_end], sr=sr)
#
#     plt.title(f'Waveform Segment {i + 1}')
#     plt.tight_layout()
#
#     # 去掉边框
#     ax = plt.gca()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     plt.show()