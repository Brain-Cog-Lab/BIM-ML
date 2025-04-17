import os
import logging
from tqdm import tqdm
from concurrent import futures
import subprocess as sp
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def is_video_valid(video_path):
    """检查视频文件是否有效"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    ret, frame = cap.read()  # 读取一帧，检查是否损坏
    cap.release()
    return ret


def delete_corrupted_video(video_path):
    """删除损坏的视频文件"""
    os.remove(video_path)
    logging.info(f"Deleted corrupted video: {video_path}")


def convert(mp4_file_path):
    """将有效的视频文件转换为avi格式"""
    # 首先检查视频文件是否有效
    if not is_video_valid(mp4_file_path):
        delete_corrupted_video(mp4_file_path)
        return "Corrupted video deleted"

    avi_file_path = mp4_file_path.replace(".mp4", ".avi")

    # 使用ffmpeg转换视频
    args = ' '.join(['ffmpeg', '-y', '-loglevel', 'error',
                     '-i', f'{mp4_file_path}',  # Specify the input
                     '-c:v', 'mpeg4',
                     '-filter:v', '\"scale=min(iw\,(256*iw)/min(iw\,ih)):-1\"',
                     '-b:v', '512k',
                     f'{avi_file_path}'])

    proc = sp.Popen(args, stdout=sp.PIPE, stderr=sp.PIPE, shell=True, universal_newlines=True, encoding='ascii')
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        logging.info(f"{mp4_file_path} corrupted")
        raise ValueError(f"Video corrupted")

    return "Success"


if __name__ == "__main__":
    root_path = "/mnt/home/hexiang/kinetics-dataset/k400/train"
    mp4_file_paths = []

    # 使用os.walk遍历目录（递归遍历子目录）
    filenames = os.listdir(root_path)

    for filename in filenames:
        if filename.lower().endswith(".mp4"):
            mp4_file_paths.append(os.path.join(root_path, filename))

    print(len(mp4_file_paths))

    # # 使用多进程来进行转换操作
    # with futures.ProcessPoolExecutor(max_workers=16) as executor:
    #     future_to_video = {
    #         executor.submit(convert, mp4_file_path): mp4_file_path
    #         for mp4_file_path in mp4_file_paths
    #     }
    #     for future in tqdm(futures.as_completed(future_to_video), total=len(mp4_file_paths)):
    #         future.result()
