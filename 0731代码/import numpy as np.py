import numpy as np
import matplotlib.pyplot as plt

# 固定随机种子
np.random.seed(42)

# 参数设置
m, n = 50, 40    # 原始矩阵大小
rank_true = 20   # 原始矩阵实际的秩（可调节）
rank_approx = 5  # 截断保留的秩（用于近似）

# Step 1: 构造一个低秩矩阵 A = U_true @ V_true
U_true = np.random.randn(m, rank_true)
V_true = np.random.randn(rank_true, n)
A = U_true @ V_true

# Step 2: 对 A 做 SVD 分解
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # S 为奇异值向量，Vt 是 V 的转置

# Step 3: 截断前 rank_approx 个奇异值
U_r = U[:, :rank_approx]
S_r = np.diag(S[:rank_approx])
Vt_r = Vt[:rank_approx, :]

# Step 4: 重构近似矩阵 A_r
A_r = U_r @ S_r @ Vt_r

# Step 5: 计算误差
fro_error = np.linalg.norm(A - A_r, ord='fro')
fro_error_rel = fro_error / np.linalg.norm(A, ord='fro')

print(f"原始矩阵大小: {A.shape}")
print(f"真实秩: {rank_true}")
print(f"截断秩（近似秩）: {rank_approx}")
print(f"Frobenius 误差: {fro_error:.4f}")
print(f"相对误差: {fro_error_rel:.4%}")

# Step 6: 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(A, cmap='viridis')
plt.title("原始矩阵 A")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(A_r, cmap='viridis')
plt.title(f"截断 SVD 近似矩阵 A_r (rank={rank_approx})")
plt.colorbar()

plt.tight_layout()
plt.show()
