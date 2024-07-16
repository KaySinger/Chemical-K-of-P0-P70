import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设定正态分布的参数
mu = 35.5  # 假设均值在中间的物质P35和P36之间
sigma = 10  # 标准差

x_values = np.arange(1, 71)

# 计算正态分布的概率密度函数
concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# 调整浓度数据，确保非负并且总和为一个固定值（例如1.0）
total_concentration = sum(concentrations)
concentrations /= total_concentration
concentrations = concentrations * 10

# 创建字典来存储浓度数据
norm_concentration_p = {f'P{i}': c for i, c in enumerate(concentrations, start=1)}
print(norm_concentration_p)

# 定义微分方程
def equations(p, t, k_values):
    k = k_values[:70]
    k_inv = k_values[70:]
    dpdt = [0] * 71
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1]**2
    dpdt[2] = k[1] * p[1]**2 + k_inv[1] * p[3] - k_inv[0] * p[2] - k[2] * p[2]**2
    for i in range(3, 70):
        dpdt[i] = k[i-1] * p[i-1]**2 + k_inv[i-1] * p[i+1] - k_inv[i-2] * p[i] - k[i] * p[i]**2
    dpdt[70] = k[39] * p[39]**2 - k_inv[38] * p[40]

    return dpdt

# 定义目标函数，用于拟合最终浓度
def objective(k):
    initial_conditions = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.linspace(0, 1000, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]
    target_concentrations = [0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 初始K值猜
initial_guess = [1, 0.93, 0.48, 1.14, 1.07, 1.69, 1.46, 2.42, 1.97, 1, 2, 1.43, 1.37, 1.32, 1.16, 0.78, 1.58, 1.44,
 2.14, 0.88, 1.60, 1.41, 1.35, 1.50, 1.07, 1, 2, 1.78, 1.75, 0.80, 1, 1.51, 1.62,  1.12, 1.71, 2.36, 1.59, 1.09, 0.60, 0.75,
0.43, 1.91, 2.15, 1.40, 0.69, 0.84, 1.09, 0.37, 0.70, 1.29, 0.84, 0.18, 1.01, 0.76, 1.12, 1.23, 0.97, 0.212, 1.35, 0.95,
0.81, 1.35, 0.61, 0.36, 1.66, 1.19, 1.67, 1.22, 0.20, 0.816, 0.27, 0.18, 0.24, 0.279, 0.77, 0.33, 0.403, 0.147, 0.616, 0.58,
0.27, 1.26, 0.63, 0.263, 0.359, 0.80, 0.115, 1.24, 0.314, 0.67, 0.539, 0.234, 1.11, 0.12, 0.52, 0.749, 0.87, 0.70, 2.46, 0.66,
0.96, 0.33, 2.37, 1.18, 1, 1.25, 0.25, 0.52, 0.549, 1.3, 2.54, 0.878, 2.13, 0.223, 1.32, 0.33, 1.41, 1.4, 1.46, 2.69, 1.43, 1, 0.879, 0.43,
0.89, 0.93, 0.38, 1.63, 0.62, 0.602, 1.01, 0.958, 0.87, 0.88, 0.17, 0.50, 0.87, 2.17, 0.84]

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * 139

result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
k_optimized = result.x

print(f'k = {k_optimized[:40]}')
print(f'k_inv = {k_optimized[40:]}')

# 利用优化后的参数进行模拟
initial_conditions = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

# 绘制浓度曲线
plt.xlabel("P-concentrations")
plt.ylabel("concentration")
plt.title("Normal distribution of Concentrations")
plt.xticks(x_values)
plt.plot(x_values, concentrations, marker='o', linestyle='-')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (20, 6))
for i in range(71):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(20, 6))
final_concentrations = sol[-1, 1:]
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65', 'p66', 'p67', 'p68', 'p69', 'p70']
plt.plot(labels, final_concentrations, 'o-', label = 'Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()