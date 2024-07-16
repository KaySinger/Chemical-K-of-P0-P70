import numpy as np

# 设定正态分布的参数
mu = 35.5  # 假设均值在中间的物质P35和P36之间
sigma = 20  # 标准差

x_values = np.arange(1, 71)

# 计算正态分布的概率密度函数
concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# 调整浓度数据，确保非负并且总和为一个固定值（例如1.0）
total_concentration = sum(concentrations)
concentrations /= total_concentration
concentrations = concentrations * 10

# 已知的 p 值
p = concentrations
print(p)

# 初始化 k 和 k_inv 数组
k = np.zeros(70)
k_inv = np.zeros(69)

# 已知的 k[0] 值
k[0] = 1
k[1] = 1
k[2] = 2
k_inv[0] = 0.000001

# 利用 dpdt[0] = 0 求解 k_inv[0]
k_inv[1] = k[2] * p[1]**2 / p[2]

# 利用中间的方程求解 k 和 k_inv
for i in range(3, 70):
    k[i] = k[i-1] * p[i-2]**2 / p[i-1]**2
    k_inv[i-1] = k_inv[i-2] * p[i-1] / p[i]

# 输出结果
print("k 数组:", k)
print("k_inv 数组:", k_inv)