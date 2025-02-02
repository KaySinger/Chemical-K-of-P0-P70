import time

import numpy as np
import math
from scipy.optimize import minimize, curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#用于显示正常中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#用于正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 41)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values

# 初始化 k 和 k_inv 数组
def initialize_k_values(concentrations):
    k = np.zeros(40)
    k_inv = np.zeros(39)
    k[0] = 1
    for i in range(1, 40):
        k[i] = 0.5 * math.log(2**i)
    for i in range(0, 39):
        k_inv[i] = k[i+1] * concentrations[i]**2 / concentrations[i+1]
    return list(k) + list(k_inv)

def correct_k_values(k, k_inv, concentrations):
    k_adjusted = sorted(k[1:])
    k_inv_adjusted = k_inv

    for i in range(39):
        k_inv_adjusted[i] = k_adjusted[i] * concentrations[i]**2 / concentrations[i+1]

    k_adjusted.insert(0, 2)

    return list(k_adjusted) + list(k_inv_adjusted)

# 定义微分方程
def equations(p, t, k_values):
    k = k_values[:40]
    k_inv = k_values[40:]
    dpdt = [0] * 41
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1]**2
    for i in range(2, 40):
        dpdt[i] = k[i-1] * p[i-1]**2 + k_inv[i-1] * p[i+1] - k_inv[i-2] * p[i] - k[i] * p[i]**2
    dpdt[40] = k[39] * p[39] ** 2 - k_inv[38] * p[40]
    return dpdt

# 定义目标函数
def objective(k):
    initial_conditions = [10] + [0] * 40
    t = np.linspace(0, 500, 500)
    sol = odeint(equations, initial_conditions, t, args=(k,))
    final_concentrations = sol[-1, :]  # 忽略 p0 和 w
    target_concentrations = [0] + list(concentrations)
    return np.sum((final_concentrations - target_concentrations) ** 2)

# 回调函数
def callback(xk):
    current_value = objective(xk)
    objective_values.append(current_value)
    if len(objective_values) > 1:
        change = np.abs(objective_values[-1] - objective_values[-2])
        print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}")

# 绘图函数
def plot_concentration_curves(t, sol):
    plt.figure(figsize=(20, 10))
    plt.plot(t, sol[:, 0], label='p0')
    for i in range(1, 11):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P0-P10 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(11, 21):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P11-P20 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(21, 31):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P21-P30 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(31, 41):
        plt.plot(t, sol[:, i], label=f'p{i}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P31-P40 Concentration over Time')
    plt.grid(True)
    plt.show()

# 模拟正态分布
mu = 20.5
sigma = 10
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

# 初始K值猜测
k_initial = np.zeros(40)
k_inv_initial = np.zeros(39)
k_initial[0] = 2
for i in range(1, 40):
    k_initial[i] = 0.5 + 0.5 * i
k_inv_initial = [0.5] * 10 + [1.5] * 5 + [2.5] * 5 + [3.5] * 5 + [2] * 5 + [1.5] * 5 + [1] * 4
initial_guess = initialize_k_values(concentrations)

# 添加参数约束，确保所有k值都是非负的
bounds = [(0, 100)] * 40 + [(0, 10)] * 39  # 确保长度为 79

# 记录目标函数值
objective_values = []

# 第一次优化
result_first = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
k_optimized = result_first.x
final_precision = result_first.fun
print("第一次优化的精度", final_precision)

# 如果第一次优化不理想，进行二次优化
if result_first.fun > 1e-08:
    # 对优化不理想的k值进行修正操作
    initial_guess = correct_k_values(k_optimized[:40], k_optimized[40:], concentrations)
    print("修正后的k值", initial_guess)
    for i in range(50):
        if final_precision > 1e-08:
            print(f"第{i + 1}次优化不理想，进行第{i + 2}次优化。")
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
            k_optimized = result.x
            final_precision = result.fun
            print(f"第{i + 2}次优化的最终精度{final_precision}")
            initial_guess = k_optimized
        else:
            break

print("最终优化的精度", final_precision)

# 输出优化结果
k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:40], start=0)}
k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_optimized[40:], start=1)}
print("优化后的k:", k_result)
print("优化后的k_inv:", k_inv_result)

# 利用优化后的参数进行模拟
initial_conditions = [10] + [0] * 40
t = np.linspace(0, 500, 500)
sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

Deviation = [0] * 40
Error = [0] * 40
p = list(concentrations)
for i in range(40):
    Deviation[i] = p[i] - sol[-1][i+1]
    if p[i] != 0:
        Error[i] = Deviation[i] / p[i]
    else:
        Error[i] = float('inf')

deviations = {f'P{i}': c for i, c in enumerate(Deviation, start=1)}
Error_Ratio = {f'Error Ratio P{i}': c for i, c in enumerate(Error, start=1)}
print("P1-P70理想最终浓度和实际最终浓度的差值是:", deviations)
print("P1-P70实际浓度与理想浓度的误差比值是:", Error_Ratio)

x_values = [f'P{i}' for i in range(1, 41)]

# 系数和聚合物状态的曲线拟合
pm = [0] * 39
for i in range(39):
   pm[i] = math.log(2**(i+1))

# 假设的模型：k = a * P^b
def model(P, a, x):
    return a * P**x

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(model, pm[:39], np.log(k_optimized[1: 40]), maxfev= 1000)

# 拟合得到的参数
a, x = popt
print(f"正向系数拟合参数: a = {a}, x = {x}")

# 使用拟合参数绘制拟合曲线
P_fit = np.linspace(min(pm[: 39]), max(pm[: 39]), 100)
k_fit = model(P_fit, *popt)

# 绘制原始数据和拟合曲线
plt.scatter(pm[: 39], np.log(k_optimized[1: 40]), label='Natural data')
plt.plot(P_fit, k_fit, color='red', label=f'curve_fitting : ln(k) = {a:.2f} * ln(2^n)^{x:.2f}(1<=n<=40)')
plt.xlabel('polymer')
plt.ylabel('ln(k)')
plt.legend()
plt.title('ln(k) vs polymer Curve_fitting')
plt.show()

# 绘制理想稳态浓度曲线
plt.figure(figsize=(20, 10))
plt.xlabel("P-Species")
plt.ylabel("P-Concentrations")
plt.title("Ideal Concentrations and Actual Concentrations")
plt.xticks(range(len(x_values)), x_values, rotation=90)
final_concentrations = sol[-1, 1:]
plt.plot(range(len(x_values)), concentrations, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
plt.plot(range(len(x_values)), final_concentrations, label = 'Actual Concentrations', marker='o', linestyle='-', color='red')
plt.grid(True)
plt.show()

# 绘制各个物质的浓度变化曲线
plot_concentration_curves(t, sol)

# 优化k值后P1-P70实际浓度与理想浓度的误差比值
plt.figure(figsize=(20, 10))
plt.xlabel("P-Species")
plt.ylabel("P-Error-Ratio")
plt.title("Error Ratio of Concentrations between Ideal and Actual")
plt.xticks(range(len(x_values)), x_values, rotation=90)
plt.plot(range(len(x_values)), Error, label = 'Error-Ratio', marker='o', linestyle='-', color='blue')
plt.grid(True)
plt.show()