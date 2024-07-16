import numpy as np
import matplotlib.pyplot as plt

# 设定正态分布的参数
mu = 35.5  # 假设均值在中间的物质P35和P36之间
sigma = 20  # 标准差

x_values = np.arange(1, 71)

# 计算正态分布的概率密度函数
concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
# print(concentrations)
# 调整浓度数据，确保非负并且总和为一个固定值（例如1.0）
total_concentration = sum(concentrations)
concentrations /= total_concentration
concentrations = concentrations * 100

# 创建字典来存储浓度数据
norm_concentration_p = {f'P{i}': c for i, c in enumerate(concentrations, start=1)}
print(norm_concentration_p, sum(concentrations), [0] + list(concentrations))
 
# 绘制浓度曲线
plt.xlabel("P-concentrations")
plt.ylabel("concentration")
plt.title("Normal distribution of Concentrations")
plt.xticks(x_values)
plt.plot(x_values, concentrations, marker='o', linestyle='-')
plt.grid(True)
plt.show()
