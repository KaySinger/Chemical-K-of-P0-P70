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

for degree in range(4, 6):
    # 初始K值猜测
    initial_guess = Optimize_K_Model.initialize_k_values(concentrations)
    k = initial_guess[:70]
    k_inv = initial_guess[70:]
    k = add_perturbation_to_k(k, perturbation_factor=0.5 * degree)
    initial_guess = list(k) + list(k_inv)
    print("随机噪声干扰后的初始值猜测:", initial_guess)

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
        for i in range(10):
            print(f"第{i+1}次优化不理想，进行第{i+2}次优化。")
            result = minimize(Optimize_K_Model.objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=Optimize_K_Model.callback)
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
    print("初始值被干扰后优化的k", k_result)
    print("初始值被干扰后优化的k_inv", k_inv_result)

    # 利用优化后的参数进行模拟
    initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
    t = np.linspace(0, 10000, 5000)
    sol = odeint(Optimize_K_Model.equations, initial_conditions, t, args=(k_optimized,))

    Deviation = [0] * 72
    p = [0, 0] + list(concentrations)
    for i in range(72):
        Deviation[i] = p[i] - sol[-1][i]

    deviations = {f'P{i}': c for i, c in enumerate(Deviation[2:], start=1)}
    print("P1-P70理想最终浓度和实际最终浓度的差值是", deviations)

    # 绘制浓度曲线
    plt.figure(figsize=(50, 20))
    plt.xlabel("P-concentrations")
    plt.ylabel("concentration")
    plt.title("Normal distribution of Concentrations")
    plt.xticks(x_values)
    plt.plot(x_values, concentrations, marker='o', linestyle='-')
    plt.grid(True)
    plt.show()

    # 绘制各个物质的浓度变化曲线
    Optimize_K_Model.plot_concentration_curves(t, sol)

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