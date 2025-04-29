# ---------------1, 清除没有的数据---------------
# import os
#
# # 文件路径
# test_file_path = "/mnt/home/hexiang/Brain-inspired mechanisms/data/KineticSound/my_train.txt"
# audio_base_path = "/mnt/home/hexiang/kinetics_sound/train/audio/"
#
# # 获取所有 .wav 文件的名字（不含扩展名）
# wav_files = set()
# for root, _, files in os.walk(audio_base_path):
#     for file in files:
#         if file.endswith(".wav"):
#             wav_files.add(file[:11])  # 只保留文件名
#
# # 读取 my_test.txt 并获取 vid_start_end
# with open(test_file_path, "r") as f:
#     lines = f.readlines()
#
# # 过滤掉不在 .wav 文件列表中的条目
# filtered_lines = [line for line in lines if line.split(",")[0][:11] in wav_files]
#
# # 更新 my_test.txt
# with open(test_file_path, "w") as f:
#     f.writelines(filtered_lines)
#
# print(f"文件更新完成，原始 {len(lines)} 条，保留 {len(filtered_lines)} 条有效数据。")


# # ---------------2. 测试集根据my_test.txt更变名字---------------
# import os
#
#
# def rename_dirs_keep_first_11(root_dir):
#     """
#     遍历 root_dir 下的所有子目录，将目录名截断为前 11 个字符。
#     """
#     # 为了从底向上处理（防止上层目录先改名导致子目录路径出问题），topdown=False
#     for parent, dirs, files in os.walk(root_dir, topdown=False):
#         for d in dirs:
#             old_dir_path = os.path.join(parent, d)
#             # 如果你的目录确实都是前 11 位有意义，可以加一个判断：只有长度 > 11 时才截断
#             if len(d) > 11:
#                 new_name = d[:11]
#                 new_dir_path = os.path.join(parent, new_name)
#
#                 # 如果 new_dir_path 不存在，才能安全地重命名
#                 if not os.path.exists(new_dir_path):
#                     print(f"Renaming: {old_dir_path} -> {new_dir_path}")
#                     os.rename(old_dir_path, new_dir_path)
#                 else:
#                     print(f"目标目录 {new_dir_path} 已存在，无法重命名，请手动处理。")
#
#
# def rename_wav_files(root_dir):
#     """
#     遍历 root_dir 下的所有子目录，将 .wav 文件的文件名(不包含扩展名的部分)截断为前 11 个字符，保留原扩展名.
#     即: 原文件名 'abcdefghijkXYZ.wav' -> 'abcdefghijk.wav'
#     如果你想完全去掉 .wav 后缀，只保留前 11 个字符，请在下面改为 new_name = old_name[:11]。
#     """
#     for parent, dirs, files in os.walk(root_dir, topdown=False):
#         for f in files:
#             # 仅处理 .wav 文件（大小写都考虑）
#             if f.lower().endswith('.wav'):
#                 old_file_path = os.path.join(parent, f)
#
#                 # 分离文件名与扩展名
#                 old_name, ext = os.path.splitext(f)  # old_name=纯文件名, ext=.wav
#
#                 # 如果文件名超过 11 个字符，才截断
#                 if len(old_name) > 11:
#                     # 保留前 11 个字符 + 原扩展名
#                     new_name = old_name[:11] + ext
#
#                     new_file_path = os.path.join(parent, new_name)
#
#                     # 检查目标文件是否已存在，避免冲突
#                     if not os.path.exists(new_file_path):
#                         print(f"Renaming file:\n  {old_file_path}\n       -> {new_file_path}\n")
#                         os.rename(old_file_path, new_file_path)
#                     else:
#                         print(f"目标文件 {new_file_path} 已存在，无法重命名，请手动处理。")
#
# if __name__ == "__main__":
#     # 假设我们只需要处理 kinetics_sound/test 目录
#     root_dir = "/mnt/home/hexiang/kinetics_sound/test/audio/"
#     # rename_dirs_keep_first_11(root_dir)
#     rename_wav_files(root_dir)


#---------------- 3. 删除过短的音频文件-------------
import os
import torchaudio
import torch
from tqdm import tqdm  # pip install tqdm

def remove_short_wav_files(root_dir, target_sr=22050, repeat_factor=3):
    """
    递归遍历 root_dir 下的所有 WAV 文件：
      1) 读取并重采样到 target_sr（默认22050）。
      2) 波形 repeat repeat_factor 次（默认3）。
      3) 若波形长度仍小于 target_sr * repeat_factor，则删除该文件。
    """
    min_length = target_sr * repeat_factor  # 22050 * 3 = 66150

    # 先收集所有待处理的 wav 文件
    wav_files = []
    for parent, dirs, files in os.walk(root_dir, topdown=False):
        for filename in files:
            if filename.lower().endswith('.wav'):
                wav_files.append(os.path.join(parent, filename))

    cnt = 0

    # 用 tqdm 显示进度
    for wav_path in tqdm(wav_files, desc="Processing WAV files", unit="file"):
        # 尝试读取 WAV 文件
        try:
            waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
        except Exception as e:
            print(f"[读取失败] {wav_path}，错误信息：{e}。删除该文件。")
            # os.remove(wav_path)  # 如果想直接删除可取消注释
            cnt += 1
            continue

        # 重采样到 target_sr
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=target_sr
            )

        # repeat
        waveform = waveform.repeat(1, repeat_factor)

        # 检查长度是否满足要求
        if waveform.shape[1] < min_length:
            print(f"[过短文件] {wav_path}，长度 {waveform.shape[1]} < {min_length}，删除。")
            # os.remove(wav_path)  # 如果想直接删除可取消注释
            cnt += 1

    print("file num:{}".format(cnt))

if __name__ == "__main__":
    # 假设我们只需要处理 kinetics_sound/test 目录下的 audio
    root_dir = "/mnt/home/hexiang/kinetics_sound/train/audio"
    remove_short_wav_files(root_dir)
