import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 该代码以ABCDE五个物质的五个反应式进行模拟，包含了正向反应和逆向反应。

# 定义微分方程
def equations(p, t, k_values):
    A, B, C, D, E = p
    k1, k2, k3, k4, k2_inv, k3_inv, k4_inv = k_values
    dAdt = - k1 * A
    dBdt = k1 * A + k2_inv * C - k2 * B
    dCdt = k2 * B + k3_inv * D - k2_inv * C - k3 * C
    dDdt = k3 * C + k4_inv * E - k3_inv * D - k4 * D
    dEdt = k4 * D - k4_inv * E

    return dAdt, dBdt, dCdt, dDdt, dEdt

# 定义目标函数，用于拟合最终浓度
def objective(k):
    initial_conditions = [1, 0, 0, 0, 0]
    t = np.linspace(0, 100, 1000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]
    target_concentrations = [0, 0.2, 0.3, 0.3, 0.2]
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 初始K值猜测
initial_guess = [1, 1, 1, 1, 1, 1, 1]

result = minimize(objective, initial_guess, method='Nelder-Mead')
k_optimized = result.x

print("optimized coefficients:", k_optimized)

# 利用优化后的参数进行模拟
initial_conditions = [1, 0, 0, 0, 0]
t = np.linspace(0, 100, 1000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
plt.plot(t, sol[:, 0], label = 'A')
plt.plot(t, sol[:, 1], label = 'B')
plt.plot(t, sol[:, 2], label = 'C')
plt.plot(t, sol[:, 3], label = 'D')
plt.plot(t, sol[:, 4], label = 'E')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(10, 6))
final_concentrations = sol[-1, 1:]
labels = ['B', 'C', 'D', 'E']
plt.plot(labels, final_concentrations, 'o-', label = 'Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()