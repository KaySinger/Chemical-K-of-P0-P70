import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 设定正态分布的参数
mu = 35.5  # 假设均值在中间的物质P35和P36之间
sigma = 20  # 标准差

x_values = np.arange(1, 71)

# 计算正态分布的概率密度函数
concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# 调整浓度数据，确保非负并且总和为一个固定值（例如1.0）
total_concentration = sum(concentrations)
concentrations /= total_concentration
concentrations = concentrations * 100

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
    for i in range(2, 70):
        dpdt[i] = k[i-1] * p[i-1]**2 + k_inv[i-1] * p[i+1] - k_inv[i-2] * p[i] - k[i] * p[i]**2
    dpdt[70] = k[69] * p[69]**2 - k_inv[68] * p[70]

    return dpdt

# 定义目标函数，用于拟合最终浓度
def objective(k):
    initial_conditions = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.linspace(0, 1000, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]
    target_concentrations = [0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 初始K值猜
initial_guess = [2, 2, 1.68732963, 1.43067617, 1.21914181, 1.04409155, 0.89865793,
 0.77735914, 0.67580357, 0.59046033, 0.51848052, 0.45755745, 0.40581706,
 0.36173159, 0.3240515,  0.29175151, 0.26398769, 0.24006326, 0.2194013,
 0.20152279, 0.18602898, 0.17258717, 0.16091921, 0.15079216, 0.14201071,
 0.13441103, 0.12785572, 0.12222975, 0.11743705, 0.11339785, 0.11004644,
 0.10732938, 0.10520412, 0.10363783, 0.10260662, 0.10209487, 0.10209487,
 0.10260662, 0.10363783, 0.10520412, 0.10732938, 0.11004644, 0.11339785,
 0.11743705, 0.12222975, 0.12785572, 0.13441103, 0.14201071, 0.15079216,
 0.16091921, 0.17258717, 0.18602898, 0.20152279, 0.2194013,  0.24006326,
 0.26398769, 0.29175151, 0.3240515,  0.36173159, 0.40581706, 0.45755745,
 0.51848052, 0.59046033, 0.67580357, 0.77735914, 0.89865793, 1.04409155, 1.21914181, 1.43067617, 1.68732963,
 0.89970359, 0.82845736, 0.76476253, 0.70773191, 0.65659367, 0.61067528,
 0.56938986, 0.53222448, 0.49873025, 0.46851371, 0.4412296,  0.41657452,
 0.3942816,  0.3741158,  0.35586995, 0.33936131, 0.32442856, 0.31092924,
 0.29873753, 0.28774232, 0.27784554, 0.26896072, 0.26101173, 0.25393171,
 0.24766211, 0.24215194, 0.23735701, 0.23323939, 0.22976691, 0.2269127,
 0.22465488, 0.22297627, 0.22186417, 0.22131021, 0.22131021, 0.22186417,
 0.22297627, 0.22465488, 0.2269127,  0.22976691, 0.23323939, 0.23735701,
 0.24215194, 0.24766211, 0.25393171, 0.26101173, 0.26896072, 0.27784554,
 0.28774232, 0.29873753, 0.31092924, 0.32442856, 0.33936131, 0.35586995,
 0.3741158,  0.3942816,  0.41657452, 0.4412296,  0.46851371, 0.49873025,
 0.53222448, 0.56938986, 0.61067528, 0.65659367, 0.70773191, 0.76476253,
 0.82845736, 0.89970359, 0.97952265]

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * 139

result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
k_optimized = result.x

print(f'k = {k_optimized[:70]}')
print(f'k_inv = {k_optimized[70:]}')

# 利用优化后的参数进行模拟
initial_conditions = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 1000, 1000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

# 检验模块，验证动态平衡时dpdt是否等于0
k_values = np.array(k_optimized)
p_values = np.array(sol[-1])

dpdt_values = equations(p_values, 0, k_values)

# Check for near-zero values in dpdt
for i, dpdt in enumerate(dpdt_values):
    print(f"dpdt[{i}] = {dpdt}")

# 绘制浓度曲线
plt.xlabel("P-concentrations")
plt.ylabel("concentration")
plt.title("Normal distribution of Concentrations")
plt.xticks(x_values)
plt.plot(x_values, concentrations, marker='o', linestyle='-')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(71):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(10, 6))
final_concentrations = sol[-1, 1:]
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65', 'p66', 'p67', 'p68', 'p69', 'p70']
plt.plot(labels, final_concentrations, 'o-', label = 'Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()