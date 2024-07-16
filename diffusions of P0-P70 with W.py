import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import initial_k_guess_with_w # 正态分布及初始k值猜测


concentrations = initial_k_guess_with_w.concentrations

# 创建字典来存储浓度数据
norm_concentration_p = {f'P{i}': c for i, c in enumerate(concentrations, start=1)}
print("动态平衡时的各物质浓度", norm_concentration_p)
# 定义微分方程
def equations(p, t, k_values):
    k = k_values[:70]
    k_inv = k_values[70:]
    dpdt = [0] * 72
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = - k[1] * p[1] * p[2] # 其中W物质设为p[1],dpdt[1] = dwdt
    dpdt[2] = k[0] * p[0] + k_inv[0] * p[3] - k[1] * p[1] * p[2]
    dpdt[3] = 2 * k[1] * p[1] * p[2] + k_inv[1] * p[4] - k_inv[0] * p[3] - k[2] * p[3]**2
    for i in range(4, 71):
        dpdt[i] = k[i - 2] * p[i - 1] ** 2 + k_inv[i - 2] * p[i + 1] - k_inv[i - 3] * p[i] - k[i-1] * p[i] ** 2 # dp3dt - dp69dt格式类似
    dpdt[71] = k[69] * p[70] ** 2 - k_inv[68] * p[71]

    return dpdt

# 定义目标函数，用于拟合最终浓度
def objective(k):
    initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.linspace(0, 1000, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]  # 忽略 p0 和 w
    target_concentrations = [0, 0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 初始K值猜测
initial_guess = list(initial_k_guess_with_w.k) + list(initial_k_guess_with_w.k_inv)

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * 139

result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
k_optimized = result.x

print("k", k_optimized[:70])
print("k_inv", k_optimized[70:])

# 利用优化后的参数进行模拟
initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
plt.figure(figsize=(50, 6))
plt.xlabel("P-concentrations")
plt.ylabel("concentration")
plt.title("Normal distribution of Concentrations")
plt.xticks(initial_k_guess_with_w.x_values)
plt.plot(initial_k_guess_with_w.x_values, concentrations, marker='o', linestyle='-')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize=(50, 6))
plt.plot(t, sol[:, 0], label='p0')
plt.plot(t, sol[:, 1], label='w')
for i in range(2, 72):
    plt.plot(t, sol[:, i], label=f'p{i-1}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(50, 6))
final_concentrations = sol[-1, 2:]
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65', 'p66', 'p67', 'p68', 'p69', 'p70']
plt.plot(labels, final_concentrations, 'o-', label='Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()