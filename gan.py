import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置超参数
batch_size = 64
z_dim = 100  # 生成器输入的随机噪声维度
lr = 0.0002
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载 (MNIST 手写数字)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root="/mnt/data/datasets", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Tanh()  # 输出范围 [-1, 1]，匹配数据的归一化
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)  # 变形为图片格式


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出为概率
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))  # 变形为一维向量


# 初始化模型
G = Generator().to(device)
D = Discriminator().to(device)

# 损失函数 & 优化器
criterion = nn.BCELoss()  # 交叉熵损失
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# 训练
for epoch in range(epochs):
    for real_images, _ in dataloader:
        real_images = real_images.to(device)

        # --------------------------
        # 1. 训练判别器 D
        # --------------------------
        optimizer_D.zero_grad()

        # 真实数据标签为 1
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)

        # 判别真实数据
        real_preds = D(real_images)
        loss_real = criterion(real_preds, real_labels)

        # 判别生成的数据
        z = torch.randn(real_images.size(0), z_dim).to(device)
        fake_images = G(z)
        fake_preds = D(fake_images.detach())  # `detach()` 避免更新 G
        loss_fake = criterion(fake_preds, fake_labels)

        # 总损失
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # --------------------------
        # 2. 训练生成器 G
        # --------------------------
        optimizer_G.zero_grad()

        # 让 D 误认为生成数据是真实数据（目标是 1）
        fake_preds = D(fake_images)
        loss_G = criterion(fake_preds, real_labels)
        loss_G.backward()
        optimizer_G.step()

    # 打印损失
    print(f"Epoch [{epoch + 1}/{epochs}], D_loss: {loss_D.item():.4f}, G_loss: {loss_G.item():.4f}")

    # 每 10 轮生成一张图片
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, z_dim).to(device)
            fake_images = G(z).cpu()
            plt.figure(figsize=(4, 4))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(fake_images[i][0], cmap="gray")
                plt.axis("off")
            plt.show()
