import os
import shutil

# 定义 A 目录和 B 目录的路径
A_dir = "/mnt/home/hexiang/kinetics-dataset/k400/train"
B_dir = "/mnt/home/hexiang/kinetics-dataset/k400/replacement"

# 获取 A 目录和 B 目录中的文件名（忽略扩展名）
A_files = {os.path.splitext(f)[0] for f in os.listdir(A_dir) if f.lower().endswith(".avi")}
# A_files = {"new.avi"}
B_files = {os.path.splitext(f)[0] for f in os.listdir(B_dir) if f.lower().endswith(".avi")}

# 找到 B 目录中有但 A 目录中缺失的文件
missing_files = B_files - A_files

print(f"A 目录中的文件数: {len(A_files)}")
print(f"B 目录中的文件数: {len(B_files)}")
print(f"需要从 B 复制到 A 的文件数: {len(missing_files)}")

# 复制缺失的文件到 A 目录
for file in missing_files:
    src_path = os.path.join(B_dir, file + ".avi")
    dst_path = os.path.join(A_dir, file + ".avi")
    if os.path.exists(src_path):  # 确保源文件存在
        shutil.copy2(src_path, dst_path)  # 复制文件并保持元数据
        print(f"已复制: {src_path} -> {dst_path}")

print("文件同步完成！")
