import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义图像的转换操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练集
trainset = torchvision.datasets.CIFAR10(root='/home/hexiang/data/datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 显示图像的函数，并调整图像大小
def imshow(img, size=(10, 10)):
    img = img / 2 + 0.5     # 反归一化
    npimg = img.numpy()
    plt.figure(figsize=size)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

# 过滤出类别为'狗'的图片
dog_images = []
for images, labels in trainloader:
    # 选择标签为'dog'的图像
    mask = labels == classes.index('dog')
    dog_images.extend(images[mask])
    if len(dog_images) >= 4:
        break

# 取前4张狗的图像
dog_images = dog_images[:4]
# 转换为网格图像并增加显示尺寸
imshow(torchvision.utils.make_grid(dog_images), size=(10, 10))
