# import os
# import numpy as np
# import h5py
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
# from functools import partial
# from tqdm import tqdm
#
# # 输入输出目录
# NPY_DIR = '/mnt/home/hexiang/MUSIC-AVQA/feats/res18_14x14/'
# H5_DIR = '/home/hexiang/MUSIC-AVQA/feats_hdf5/res18_14x14/'
# os.makedirs(H5_DIR, exist_ok=True)
#
# # 并行参数
# NUM_PROCESSES = 8  # 进程数
# NUM_THREADS = 0    # 线程数
#
# def thread_task(data_slice):
#     """
#     演示用：对传入的数据做一些处理。
#     如果数据已经在 np.load 后不需要处理，可以省略此逻辑。
#     """
#     # 在这里可以执行任意耗时或可并行的操作，如某些数值变换、简单校验等
#     # 这只是个示例，假设我们做一次简单加法
#     return data_slice + 1.0
#
# def process_single_npy(npy_dir, h5_dir, file_name, num_threads=1):
#     """
#     多进程的目标函数：处理一个 .npy 文件并输出对应的 .h5 文件。
#     如果没有多线程需求，可直接在这里完成数据的加载和保存。
#     """
#     if not file_name.endswith('.npy'):
#         return  # 跳过非 .npy 文件
#
#     # 构造路径
#     npy_path = os.path.join(npy_dir, file_name)
#     video_id = os.path.splitext(file_name)[0]
#     h5_path = os.path.join(h5_dir, video_id + '.h5')
#
#     # 如果 h5 文件已存在，则跳过
#     if os.path.exists(h5_path):
#         return
#
#     try:
#         # 1) 直接一次性读入 .npy
#         data = np.load(npy_path)  # data.shape 可任意
#
#         # 2) 如果需要对 data 做进一步的操作，可以在这里利用多线程
#         if num_threads > 1:
#             # 例如将 data 拆成若干片段（这里只是演示拆成行）
#             # 不需要 chunking 时，也可以按“处理任务”拆分，比如处理 data 的每一行/每一段
#             # 这里只演示把 data 分成每一行，然后交由线程池处理
#             results = []
#             with ThreadPoolExecutor(max_workers=num_threads) as executor:
#                 futures = []
#                 for row in data:
#                     futures.append(executor.submit(thread_task, row))
#                 for fut in as_completed(futures):
#                     results.append(fut.result())
#
#             # 将列表合并回原 shape
#             processed_data = np.array(results)
#         else:
#             # 如果不需要多线程处理，就直接用原数据
#             processed_data = data
#
#         # 3) 写入 .h5
#         with h5py.File(h5_path, 'w') as hf:
#             hf.create_dataset('features', data=processed_data)
#
#     except Exception as e:
#         print(f"[ERROR] Failed to convert {npy_path}: {e}")
#
# def main():
#     # 收集所有 .npy 文件
#     file_list = [f for f in os.listdir(NPY_DIR) if f.endswith('.npy')]
#     file_list.sort()
#
#     # 为进程池准备任务函数 + 参数
#     convert_func = partial(
#         process_single_npy,
#         NPY_DIR,
#         H5_DIR,
#         num_threads=NUM_THREADS
#     )
#
#     # 启动多进程
#     futures = []
#     with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
#         for fn in file_list:
#             futures.append(executor.submit(convert_func, fn))
#
#         # 用 tqdm 查看进度
#         for _ in tqdm(as_completed(futures), total=len(futures)):
#             pass
#
# if __name__ == "__main__":
#     main()


import json

# 假设你的JSON文件名为 "data.json"
filename = '/mnt/home/hexiang/MUSIC-AVQA/data/json_update/avqa-train.json'

# 读取JSON文件
with open(filename, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有的 "anser" 字段并存储到一个 set 中
answer_set = set(entry['anser'] for entry in data)

# 输出结果
print(answer_set)
