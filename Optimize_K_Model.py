import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 71)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values


# 初始化 k 和 k_inv 数组
def initialize_k_values(concentrations):
    k = np.zeros(70)
    k_inv = np.zeros(68)
    k[0], k[1], k[2] = 1, 1, 2
    k_inv[0] = (k[2] * concentrations[1] ** 2) / concentrations[2]
    for i in range(3, 70):
        k[i] = k[i - 1] * concentrations[i - 2] ** 2 / concentrations[i - 1] ** 2
        k_inv[i - 2] = k_inv[i - 3] * concentrations[i - 1] / concentrations[i]
    return list(k) + list(k_inv)

# 定义微分方程
def equations(p, t, k_values):
    k = k_values[:70]
    k_inv = k_values[70:]
    dpdt = [0] * 72
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = - k[1] * p[1] * p[2]
    dpdt[2] = k[0] * p[0] - k[1] * p[1] * p[2]
    dpdt[3] = 2 * k[1] * p[1] * p[2] + k_inv[0] * p[4] - k[2] * p[3] ** 2
    for i in range(4, 71):
        dpdt[i] = k[i - 2] * p[i - 1] ** 2 + k_inv[i - 3] * p[i + 1] - k_inv[i - 4] * p[i] - k[i - 1] * p[i] ** 2
    dpdt[71] = k[69] * p[70] ** 2 - k_inv[67] * p[71]
    return dpdt

# 定义目标函数
def objective(k):
    initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
    t = np.linspace(0, 10000, 5000)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]  # 忽略 p0 和 w
    target_concentrations = [0, 0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 回调函数
def callback(xk):
    current_value = objective(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}")

# 移动平滑函数
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def correct_k_values(k, k_inv):
    k_initial = k[2:]
    k_inv_initial = k_inv

    # 找到中间点
    k_mid_index = len(k_initial) // 2
    k_inv_mid_index = len(k_inv_initial) // 2

    # 前半段递减排列
    k_front = sorted(k_initial[:k_mid_index], reverse=True)
    k_inv_front = sorted(k_inv_initial[:k_inv_mid_index], reverse=True)

    # 后半段等于前半段的逆
    k_back = k_front[::-1]
    k_inv_back = k_inv_front[::-1]

    # 合并前半段和后半段
    k_adjusted = k_front + k_back
    k_adjusted = list(k[:2]) + list(k_adjusted)
    k_inv_adjusted = k_inv_front + k_inv_back

    return list(k_adjusted) + list(k_inv_adjusted)

# 绘图函数
def plot_concentration_curves(t, sol):
    plt.figure(figsize=(50, 10))
    plt.plot(t, sol[:, 0], label='p0')
    plt.plot(t, sol[:, 1], label='w')
    for i in range(2, 12):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P0-P10 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(12, 22):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P11-P20 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(22, 32):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P21-P30 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(32, 42):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P31-P40 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(42, 52):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P41-P50 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(52, 62):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P51-P60 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 10))
    for i in range(62, 72):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P61-P70 Concentration over Time')
    plt.grid(True)
    plt.show()


# 模拟正态分布
mu = 35.5
sigma = 20
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

# 初始K值猜测
initial_guess = initialize_k_values(concentrations)

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, 5)] * 70 + [(0, 0.5)] * 68  # 确保长度为 139

# 记录目标函数值
objective_values = []

# 第一次优化
result_first = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result_first.x
final_precision = result_first.fun
print(f"第一次优化的最终精度是{final_precision}")

# 如果第一次优化不理想，进行二次优化
if result_first.fun > 1e-08:
    # 对优化不理想的k值进行修正操作
    k_smoothed = moving_average(k_optimized[:70], window_size=5)
    k_inv_smoothed = moving_average(k_optimized[70:], window_size=5)
    k_optimized = list(k_smoothed) + list(k_inv_smoothed)
    initial_guess = correct_k_values(k_optimized[:70], k_inv_smoothed[:70])
    print("修正后的k值", initial_guess)
    for i in range(5):
        print(f"第{i+1}次优化不理想，进行第{i+2}次优化。")
        result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
        k_optimized = result.x
        final_precision = result.fun
        print(f"第{i+2}次优化的最终精度{final_precision}")
        initial_guess = k_optimized
        if final_precision < 1e-08:
            break

print("最终优化的精度", final_precision)

# 输出优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:70], start=0)}
k_inv_result = [0.00000001] + list(k_optimized[70:])
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_result, start=1)}
print("优化后的k", k_result)
print("k_inv", k_inv_result)

# 利用优化后的参数进行模拟
initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
t = np.linspace(0, 10000, 5000)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

Deviation = [0] * 72
Error = [0] * 72
p = [0, 0] + list(concentrations)
for i in range(72):
    Deviation[i] = p[i] - sol[-1][i]
    Error[i] = Deviation[i] / p[i]

deviations = {f'P{i}': c for i, c in enumerate(Deviation[2:], start=1)}
Error_Ratio = {f'Error Ratio of P{i}': c for i, c in enumerate(Error[2:], start=1)}
print("P1-P70理想最终浓度和实际最终浓度的差值是", deviations)
print("P1-P70优化的误差是", Error_Ratio)

# 绘制理想稳态浓度曲线
plt.figure(figsize=(50, 20))
plt.xlabel("P-concentrations")
plt.ylabel("concentration")
plt.title("Ideal Normal distribution of Concentrations")
plt.xticks(x_values)
plt.plot(x_values, concentrations, marker='o', linestyle='-')
plt.grid(True)
plt.show()

# 绘制各个物质的浓度变化曲线
plot_concentration_curves(t, sol)

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(50, 20))
final_concentrations = sol[-1, 2:]
labels = [f'p{i + 1}' for i in range(70)]
plt.plot(labels, final_concentrations, 'o-', label='Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()
