import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 生成训练数据
x = torch.unsqueeze(torch.linspace(-2 * torch.pi, 2 * torch.pi, 200), dim=1)  # [200, 1]
y = torch.sin(x)  # 目标函数 y = sin(x)

# 2. 定义 MLP 网络结构（2层全连接 + ReLU）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),     # 输入层到隐藏层
            nn.ReLU(),
            nn.Linear(64, 1)      # 隐藏层到输出层
        )

    def forward(self, x):
        return self.model(x)

# 3. 实例化模型、定义损失函数和优化器
model = MLP()
criterion = nn.MSELoss()  # 均方误差
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 梯度下降

# 4. 训练过程
for epoch in range(500):
    y_pred = model(x)  # 前向传播

    loss = criterion(y_pred, y)  # 计算损失

    optimizer.zero_grad()  # 清除旧梯度
    loss.backward()        # 反向传播，计算梯度
    optimizer.step()       # 更新参数

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 5. 可视化拟合结果
plt.plot(x.detach().numpy(), y.numpy(), label="True sin(x)")
plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label="MLP Prediction")
plt.legend()
plt.title("MLP Approximates sin(x)")
plt.show()
