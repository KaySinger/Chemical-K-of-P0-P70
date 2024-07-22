# Chemical-K-of-P0-P70
Constructing a mathematical model to estimate the optimal values of  𝑘. The reaction equations for the 72 substances, including P0-P70 and W (water), involve both forward and reverse reactions, resulting in 139 coefficients that need to be determined. 
代码解析
1、模块1——正态分布
利用正态分布公式得到动态平衡时P1-P70的浓度并调整浓度数据使得浓度总和为10。
首先设置均值为35.5，这样可以使得到的P在x轴分布在1-70。
    import numpy as np
    
    # 设定正态分布的参数
    mu = 35.5  # 假设均值在中间的物质P35和P36之间
    sigma = 20  # 标准差
    
    x_values = np.arange(1, 71)
    
    # 计算正态分布的概率密度函数
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    
    # 调整浓度数据，确保非负并且总和为一个固定值（例如1.0）
    total_concentration = sum(concentrations)
    concentrations /= total_concentration
    concentrations = concentrations * 10
    
    # 已知的 p 值
    p = concentrations
    print(p)



2、模块2——函数定义
定义两个函数，其一是定义反应式的微分方程，其二定义目标函数用于拟合最终浓度。
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





3、模块3——初始k值猜测
根据公式以及动态平衡时dpdt=0的特性猜测初始k值，这一模块主要用于物质过多时，当反应物数量过高时， 各物质浓度开始出现明显差距，需要有个更合理的初始k值猜测。根据公式可以知道有k0、k1、k1_inv的设置特殊，直接定为1、1、0.000001。
    # 初始化 k 和 k_inv 数组
    k = np.zeros(70)
    k_inv = np.zeros(69)
    
    # 已知的 k[0] 值
    k[0] = 1
    k[1] = 1
    k[2] = 2
    k_inv[0] = 0.000001
    
    # 利用 dpdt[0] = 0 求解 k_inv[0]
    k_inv[1] = k[2] * p[1]**2 / p[2]
    
    # 利用中间的方程求解 k 和 k_inv
    for i in range(3, 70):
        k[i] = k[i-1] * p[i-2]**2 / p[i-1]**2
        k_inv[i-1] = k_inv[i-2] * p[i-1] / p[i]
    
    # 输出结果
    print("k 数组:", k)
    print("k_inv 数组:", k_inv)

4、模块4——Minimize优化k值
利用python库中的minimize工具，使用Nelder-Mead估计出k值的最优解，并使用约束条件使k值不至于出现负数，符合实际化学反应机制。

5、模块5——Odeint求解微分方程
使用python库中的odeint工具，利用优化后的k值代入微分方程得到P0到P70随时间的浓度变化。
6、模块6——图像获取
使用matplotlib得到三张图，图一是模块一中利用正态分布得到的P1-P70的浓度分布曲线，图二是P1到P70各物质的浓度随时间变化的曲线，图三是反应到达动态平衡时P1-P70的最终浓度分布曲线，用于和图一做比对，验证k值是否是最优解。

7、模块7——检测机制
将优化后的k值代入公式解出动态平衡时的dpdt，用于检验反应是否达到动态平衡以及当数据出现问题时用于观测具体哪些k值优化不够好。

7个模块共分为两个py文件，其中模块1、3在initial_k_guess_with_w.py中，模块2、4、5、6、7均在diffusions of P0-P70 with W.py中，前者主要作用是数值估计，后者主要作用是调用前者的数据并且优化数据。

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
