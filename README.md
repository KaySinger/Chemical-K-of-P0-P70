# Chemical-K-of-P0-P70
本模型基于L-BFGS-B优化算法解决复杂的化学方程式系数优化问题。

对于P0-P70的72个反应方程式，其中k0-k69有70个正向反应系数，k1_inv-k69_inv有69个逆向反应系数。

已知当反应达到稳态时，P0和W物质消耗完毕，P1-P70呈现正态分布，所以直接模拟一个正态分布作为稳态浓度，

利用数学方法猜测出合理的k值并利用优化算法进行迭代后得到最优解。

# 模型的优势
1、将所有功能都分模块处理，利于维护和使用。

2、加入不良优化结果修正功能和多次优化功能，系统鲁棒性高，面对初始值不合理和噪声干扰情况，系统依旧能将系数优化到最优解。

3、通用性，可以解决不同类型的系数优化需要。

4、可检测性，加入日志功能，可以查看迭代次数和每次迭代的优化情况以及最终迭代的收敛性。
# 代码解析

# 一、模拟正态分布曲线
    # 定义正态分布函数
    def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
        x_values = np.arange(1, 71)
        concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        concentrations /= sum(concentrations)
        concentrations *= scale_factor
        return concentrations, x_values
1、利用正态分布公式定义函数。

# 二、定义微分方程
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
1、对于特殊方程单独列出，中间的相似方程使用循环。

# 三、定义目标函数和回调函数，用于拟合最终浓度
    # 定义目标函数
    def objective(k):
        initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
        t = np.linspace(0, 10000, 5000)
        sol = odeint(equations, initial_conditions, t, args=(k,))
        final_concentrations = sol[-1, :]  # 忽略 p0 和 w
        target_concentrations = [0, 0] + list(concentrations)
        return np.sum((final_concentrations - target_concentrations) ** 2)
    
    # 定义回调函数
    def callback(xk):
        current_value = objective(xk)
        objective_values.append(current_value)
        if len(objective_values) > 1:
            change = np.abs(objective_values[-1] - objective_values[-2])
            print(f"迭代次数 {len(objective_values) - 1}: 变化 = {change}")
1、定义目标函数，用于拟合最终浓度。

2、定义回调函数，用于记录迭代次数和收敛性

# 四、初始值猜测
    # 定义初始k值猜测
    def initialize_k_values(concentrations):
        k = np.zeros(70)
        k_inv = np.zeros(68)
        k[0], k[1], k[2] = 1, 1, 2
        k_inv[0] = (k[2] * concentrations[1] ** 2) / concentrations[2]
        for i in range(3, 70):
            k[i] = k[i - 1] * concentrations[i - 2] ** 2 / concentrations[i - 1] ** 2
            k_inv[i - 2] = k_inv[i - 3] * concentrations[i - 1] / concentrations[i]
        return list(k) + list(k_inv)
1、根据公式以及动态平衡时dpdt=0的特性猜测初始k值。

# 五、修正函数
    # 定义移动平滑函数
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    
    # 定义k值修正函数
    def correct_k_values(k, k_inv, window_size=5):
        # 对 k 和 k_inv 进行移动平均平滑处理
        k_smoothed = moving_average(k, window_size)
        k_inv_smoothed = moving_average(k_inv, window_size)

        k_initial = k_smoothed[2:]
        k_inv_initial = k_inv_smoothed

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
        k_adjusted = list(k_smoothed[:2]) + list(k_adjusted)
        k_inv_adjusted = k_inv_front + k_inv_back

        return list(k_adjusted) + list(k_inv_adjusted)
1、利用移动平滑函数和k值修正函数对优化不理想的k值进行修复，利于二次优化，有效解决了优化模型在第一次优化不理想后陷入局部最优解的问题，提高系统鲁棒性。

# 六、浓度变化曲线绘制函数
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
1、图像过多，以10个为一组，将浓度变化曲线分为多组，便于查看。

# 七、主函数
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
        for i in range(5):
            print(f"第{i+1}次优化不理想，进行第{i+2}次优化。")
            initial_guess = correct_k_values(k_optimized[:70], k_optimized[70:], window_size=5)
            print(f"修正后的初始值{initial_guess}")
            result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=callback)
            k_optimized = result.x
            final_precision = result.fun
            print(f"第{i+2}次优化的最终精度{final_precision}")
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
    p = [0, 0] + list(concentrations)
    for i in range(72):
        Deviation[i] = p[i] - sol[-1][i]

    deviations = {f'P{i}': c for i, c in enumerate(Deviation[2:], start=1)}
    print("P1-P70理想最终浓度和实际最终浓度的差值是", deviations)

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
利用正态分布公式得到动态平衡时P1-P70的浓度并调整浓度数据使得浓度总和为10。

1、设置均值为35.5，使P在x轴上以1-70上呈现，设置方差为20，得到的曲线更加平滑，打印理想稳态浓度分布作为参考浓度。

2、设置初始k值猜测，合理的初始k值更利于优化。

3、设置约束并开始第一次优化，倘若第一次优化不理想就会进行多次优化。

4、得到优化结果，迭代次数和优化的收敛性。

5、利用优化后的系数进行模拟，并计算实际浓度与理想浓度的差值。

6、绘制理想稳态浓度分布曲线作为参考曲线；绘制浓度变化曲线；绘制实际稳态浓度分布曲线。