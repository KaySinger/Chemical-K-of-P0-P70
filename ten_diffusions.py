import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义微分方程
def equations(p, t, k_values):
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = p
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k1_inv, k2_inv, k3_inv, k4_inv, k5_inv, k6_inv, k7_inv, k8_inv, k9_inv = k_values
    dp0dt = - k0 * p0
    dp1dt = k0 * p0 + k1_inv * p2 - k1 * p1**2
    dp2dt = k1 * p1**2 + k2_inv * p3 - k1_inv * p2 - k2 * p2**2
    dp3dt = k2 * p2**2 + k3_inv * p4 - k2_inv * p3 - k3 * p3**2
    dp4dt = k3 * p3**2 + k4_inv * p5 - k3_inv * p4 - k4 * p4**2
    dp5dt = k4 * p4**2 + k5_inv * p6 - k4_inv * p5 - k5 * p5**2
    dp6dt = k5 * p5**2 + k6_inv * p7 - k5_inv * p6 - k6 * p6**2
    dp7dt = k6 * p6**2 + k7_inv * p8 - k6_inv * p7 - k7 * p7**2
    dp8dt = k7 * p7**2 + k8_inv * p9 - k7_inv * p8 - k8 * p8**2
    dp9dt = k8 * p8**2 + k9_inv * p10 - k9_inv * p9 - k9 * p9**2
    dp10dt = k9 * p9**2 - k9_inv * p10

    return [dp0dt, dp1dt, dp2dt, dp3dt, dp4dt, dp5dt, dp6dt, dp7dt, dp8dt, dp9dt, dp10dt]

# 定义目标函数，用于拟合最终浓度
def objective(k):
    initial_conditions = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t = np.linspace(0, 100, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]
    target_concentrations = [0, 0.078, 0.091, 0.103, 0.112, 0.116, 0.116, 0.112, 0.103, 0.091, 0.078]
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 初始K值猜
initial_guess = [1] * 19

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, None)] * 19

result = minimize(objective, initial_guess, method='Nelder-Mead', bounds=bounds)
k_optimized = result.x

print("optimized coefficients:", k_optimized)

# 利用优化后的参数进行模拟
initial_conditions = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t = np.linspace(0, 100, 1000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(11):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(10, 6))
# last_row = sol[-1]
# mask = np.ones(last_row.shape, dtype=bool)
# mask[[0,2]] = False
final_concentrations = sol[-1, 1:]
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
plt.plot(labels, final_concentrations, 'o-', label = 'Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()