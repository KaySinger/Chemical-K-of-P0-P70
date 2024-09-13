import Optimize_K_Model
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize


def add_perturbation_to_k(k, perturbation_factor):
    k = np.array(k)
    perturbation = np.random.uniform(-perturbation_factor, perturbation_factor, k.shape)
    perturbed_k = k * (1 + perturbation)
    return perturbed_k

# 模拟正态分布
mu = 35.5
sigma = 20
scale_factor = 10
concentrations, x_values = Optimize_K_Model.simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})
x = [2, 20, 35, 50, 69]

for i in range(len(x)):
    # 初始K值猜测
    initial_guess = Optimize_K_Model.initialize_k_values(concentrations)
    k = initial_guess[:70]
    k[x[i]] = 5
    k_inv = initial_guess[70:]
    initial_guess = list(k) + list(k_inv)
    print("单个k值改变后的初始值猜测:", {f'k{i}': c for i, c in enumerate(k, start=0)})
    print("k_inv值保持不变:", {f'k{i}_inv': c for i, c in enumerate(k_inv, start=1)})

    # 添加参数约束，确保所有k值都是非负的
    bounds = [(0, 5)] * 70 + [(0, 0.5)] * 68  # 确保长度为 139

    # 记录目标函数值
    objective_values = []

    # 第一次优化
    result_first = minimize(Optimize_K_Model.objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=Optimize_K_Model.callback)
    k_optimized = result_first.x
    final_precision = result_first.fun
    print(f"第一次优化的最终精度是{final_precision}")

    # 如果第一次优化不理想，进行二次优化
    if result_first.fun > 1e-08:
        # 对不理想优化的k值进行修正处理
        k_smoothed = Optimize_K_Model.moving_average(k_optimized[:70], window_size=5)
        k_inv_smoothed = Optimize_K_Model.moving_average(k_optimized[70:], window_size=5)
        k_optimized = list(k_smoothed) + list(k_inv_smoothed)
        initial_guess = Optimize_K_Model.correct_k_values(k_optimized[:70], k_optimized[70:])
        print(f"修正后的初始值{initial_guess}")
        for i in range(50):
            if final_precision > 1e-08:
                print(f"第{i+1}次优化不理想，进行第{i+2}次优化。")
                result = minimize(Optimize_K_Model.objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=Optimize_K_Model.callback)
                k_optimized = result.x
                final_precision = result.fun
                print(f"第{i+2}次优化的最终精度{final_precision}")
                initial_guess = k_optimized
            else:
                break

    print("最终优化的精度", final_precision)

    # 输出优化结果
    k_result = {f"k{i}": c for i, c in enumerate(k_optimized[:70], start=0)}
    k_inv_result = [0.00000001] + list(k_optimized[70:])
    k_inv_result = {f"k{i}_inv": c for i, c in enumerate(k_inv_result, start=1)}
    print("初始值被干扰后优化的k", k_result)
    print("初始值被干扰后优化的k_inv", k_inv_result)

    # 利用优化后的参数进行模拟
    initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
    t = np.linspace(0, 10000, 5000)
    sol = odeint(Optimize_K_Model.equations, initial_conditions, t, args=(k_optimized,))

    Deviation = [0] * 70
    Error = [0] * 70
    p = list(concentrations)
    for i in range(70):
        Deviation[i] = p[i] - sol[-1][i + 2]
        if p[i] != 0:
            Error[i] = Deviation[i] / p[i]
        else:
            Error[i] = float('inf')

    deviations = {f'P{i}': c for i, c in enumerate(Deviation, start=1)}
    Error_Ratio = {f'Error Ratio of P{i}': c for i, c in enumerate(Error, start=1)}
    print("P1-P70理想最终浓度和实际最终浓度的差值是", deviations)
    print("P1-P70实际浓度与理想浓度的误差比值是", Error_Ratio)

    x_values = [f'P{i}' for i in range(1, 71)]

    # 绘制理想稳态浓度曲线
    plt.figure(figsize=(20, 10))
    plt.xlabel("P-Species")
    plt.ylabel("P-Concentrations")
    plt.title("Ideal Concentrations and Actual Concentrations")
    plt.xticks(range(len(x_values)), x_values, rotation=90)
    final_concentrations = sol[-1, 2:]
    plt.plot(range(len(x_values)), concentrations, label='Ideal Concentrations', marker='o', linestyle='-',
             color='blue')
    plt.plot(range(len(x_values)), final_concentrations, label='Actual Concentrations', marker='o', linestyle='-',
             color='red')
    plt.grid(True)
    plt.show()

    # 绘制各个物质的浓度变化曲线
    Optimize_K_Model.plot_concentration_curves(t, sol)

    # 优化k值后P1-P70实际浓度与理想浓度的误差比值
    plt.figure(figsize=(20, 10))
    plt.xlabel("P-Species")
    plt.ylabel("P-Error-Ratio")
    plt.title("Error Ratio of Concentrations between Ideal and Actual")
    plt.xticks(range(len(x_values)), x_values, rotation=90)
    plt.plot(range(len(x_values)), Error, label='Error-Ratio', marker='o', linestyle='-', color='blue')
    plt.grid(True)
    plt.show()