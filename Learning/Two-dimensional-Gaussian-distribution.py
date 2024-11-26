# 本文件用于记录学习生成对抗网络（GAN）的过程，以及实现一个简单的GAN模型。
# 在本文件中，使用PyTorch实现一个简单的GAN，用于生成二维高斯分布的数据点，
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个“生成器”类，继承了torch.nn.Module类。torch.nn.Module是PyTorch中所有神经网络模块的基类，提供了基础功能，比如参数管理和模块调用。
class Generator(nn.Module):
    # 面向对象编程中的构造函数，用于初始化对象的属性。在创建类的实例时，会自动调用init方法。
    # 这里定义了生成器需要的两个参数：input_dim：输入的特征维度（例如噪声的长度）；output_dim：输出的特征维度（例如生成数据的长度）。
    # 这个类的目的是将随机噪声经过self.net，依次通过两层全连接网络和一个激活函数，最终得到输出数据。
    def __init__(self, input_dim, output_dim):
        # super()函数是用于调用父类(超类)的一个方法。这里调用了父类的初始化方法，生成器继承了nn.Module的功能，比如参数注册和前向传播的能力。
        super(Generator, self).__init__()
        # nn.Sequential是一个容器，可以将多个层按顺序组合起来。
        # 定义生成器的神经网络结构self.net，包括两个线性层和一个ReLU激活函数。
        self.net = nn.Sequential(
            # 一个全连接层，将输入的input_dim维度的向量映射到16维。
            nn.Linear(input_dim, 16),
            # 一个激活函数，ReLU（Rectified Linear Unit，整流线性单元），将输出中所有负值置为0，保留正值不变。
            nn.ReLU(),
            # 一个全连接层，将16维的向量映射到output_dim维。
            nn.Linear(16, output_dim)
        )

    # forward方法定义了生成器的前向传播逻辑，是PyTorch模型中必须实现的关键方法。
    # 输入x（通常是随机噪声）会依次经过self.net中的所有层，最终生成输出。
    # 当你调用生成器实例（如generator(x)）时，实际上触发了forward方法。（这里是PyTorch的自动设计，在调用的时候会自动执行forward函数）
    def forward(self, x):
        return self.net(x)


# 定义一个“判别器”类，同样继承了torch.nn.Module类。
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            # LeakyReLU是一个激活函数，类似于ReLU，但是在负值区域的输出更小，可以缓解梯度消失问题。
            # 普通的ReLU 在负值区域直接输出0，可能导致神经元“死亡”（即梯度为0），导致模型的训练停滞。
            nn.LeakyReLU(0.2),
            # 一个全连接层，将隐藏层的特征（16维）的向量映射到1维。
            nn.Linear(16, 1),
            # 将标量输出从(-∞,+∞)压缩到(0,1)区间，作为一个概率值。
            # 输出值越接近1，判别器越认为输入是“真实数据”；输出值越接近0，判别器越认为输入是“生成数据”。
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 超参数
# latent_dim是生成器的输入噪声向量的维度，也称为潜在空间的维度。当latent_dim=2时，噪声向量是二维的，可以用二维平面中的点表示，便于可视化。
# 较小的latent_dim可能限制生成器的能力，因为潜在空间不足以表示复杂的数据分布；而过大的latent_dim会增加训练难度。
# 如果要生成更复杂的数据（如高分辨率图像），latent_dim通常会设置为更高。
latent_dim = 2
# data_dim是真实数据和生成数据的特征维度；在二维情况下（data_dim=2），数据可以可视化为平面上的点。
data_dim = 2
# 学习率决定了每次参数更新的步幅大小，影响模型训练的速度和稳定性。
# 生成器和判别器使用不同的学习率（如生成器的学习率稍低），以避免一个网络过强导致训练不平衡。
lr_G = 0.0004  # Learning rate
lr_D = 0.0011
# epochs是训练的轮数，每个epoch表示模型对整个训练数据集的一次完整训练。
epochs = 10000
# batch_size是每次训练时，模型从训练数据集中取出的样本数量。较大的batch_size可以提高训练速度，但可能会降低模型的泛化能力。
batch_size = 64

# 初始化生成器和判别器实例，以及优化器和损失函数。
generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)
# 优化器用于计算梯度并更新模型参数，使生成器逐步改进生成样本的质量，判别器逐步提高对真实数据和生成数据的判别能力。
# Adam是一种常用的优化算法，结合了动量(Momentum)和RMSprop的优点，适用于大多数深度学习任务。
# 动量：加速收敛，减少抖动。RMSProp：自动调整学习率，适应不同梯度大小。
# torch.optim 是 PyTorch 提供的一组优化器工具，用于更新模型的可训练参数。它的核心功能是根据计算出的梯度更新模型的参数值。
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D)
# 二分类交叉熵损失函数，用于计算生成器和判别器的损失值，衡量判别器的预测结果和真实标签之间的差异。
# BCE 是概率分布差异的经典度量，适合 GAN 的目标。
loss_fn = nn.BCELoss()


# 生成真实的高斯分布(2D Gaussian distribution)
def real_data_sampler(batch_size):
    return torch.randn(batch_size, data_dim)


# 训练循环
for epoch in range(epochs):
    # 判别器的训练
    # 真实数据由real_data_sampler生成，torch.randn生成服从标准正态分布的随机噪声。
    real_data = real_data_sampler(batch_size)
    # 同样使用torch.randn生成服从标准正态分布的随机噪声；
    noise = torch.randn(batch_size, latent_dim)
    # 将噪声输入到生成器，生成假数据。
    fake_data = generator(noise)

    # 设置标签，为真实数据分配标签 1（表示真实）；为生成数据分配标签 0（表示假数据）。
    # torch.ones生成全为1的张量，torch.zeros生成全为0的张量。作用：为数据分配标签。
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # 计算判别器的损失值，包括真实数据的损失和生成数据的损失。
    real_loss = loss_fn(discriminator(real_data), real_labels)
    # fake_data.detach()是为了防止梯度传播到生成器，因为我们只想更新判别器的参数。
    fake_loss = loss_fn(discriminator(fake_data.detach()), fake_labels)
    # d_loss：判别器损失；
    # 对真实数据（real_data），判别器输出接近 1（即 "真实"）。对生成数据（fake_data），判别器输出接近 0（即 "伪造"）。
    # 判别器总的损失值；d_loss越小，判别器越擅长区分真实数据和生成数据。
    d_loss = real_loss + fake_loss

    # 梯度清零，避免梯度累加。
    optimizer_D.zero_grad()
    # 反向传播，计算梯度。
    d_loss.backward()
    # 更新判别器的参数。
    optimizer_D.step()

    # 生成器的训练
    # 同样使用torch.randn生成服从标准正态分布的随机噪声；（这里的noise是局部变量，不存在重复定义问题）
    noise = torch.randn(batch_size, latent_dim)
    # 将噪声输入到生成器，生成假数据。
    fake_data = generator(noise)
    # g_loss：生成器损失；
    # 最小化生成数据被判别器识别为假的概率，即最大化欺骗判别器的能力。
    # 生成器的目标是让判别器将生成数据预测为真实数据，因此标签为1。
    g_loss = loss_fn(discriminator(fake_data), real_labels)

    # 梯度清零，避免梯度累加。
    optimizer_G.zero_grad()
    # 反向传播，计算梯度。
    g_loss.backward()
    # 更新生成器的参数。
    optimizer_G.step()

    # 每隔 1000 次轮次打印一次损失值，方便监控训练进展。
    # D Loss 和 G Loss 应该逐渐接近平衡，表明生成器生成的数据与真实数据的分布越来越接近，判别器无法轻易区分两者。
    # G Loss 靠近0.693；D Loss 靠近1.386。
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# Generate data and visualize
with torch.no_grad():
    noise = torch.randn(1000, latent_dim)
    generated_data = generator(noise)

real_data_sample = real_data_sampler(1000)

plt.scatter(real_data_sample[:, 0], real_data_sample[:, 1], label="Real Data", alpha=0.6)
plt.scatter(generated_data[:, 0], generated_data[:, 1], label="Generated Data", alpha=0.6)
plt.legend()
plt.show()
