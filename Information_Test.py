import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

def equations(p, t, k_values):
    k = k_values[:70]
    k_inv = k_values[70:]
    dpdt = [0] * 72
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = - k[1] * p[1] * p[2]
    dpdt[2] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1] * p[2]
    dpdt[3] = 2 * k[1] * p[1] * p[2] + k_inv[1] * p[4] - k[2] * p[3] ** 2
    for i in range(4, 71):
        dpdt[i] = k[i - 2] * p[i - 1] ** 2 + k_inv[i - 2] * p[i + 1] - k_inv[i - 3] * p[i] - k[i - 1] * p[i] ** 2
    dpdt[71] = k[69] * p[70] ** 2 - k_inv[68] * p[71]
    return dpdt

def simulate_normal_distribution(mu, sigma, total_concentration, scale_factor):
    x_values = np.arange(1, 71)
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations, x_values

# 模拟正态分布
mu = 35.5
sigma = 20
scale_factor = 10
concentrations, x_values = simulate_normal_distribution(mu, sigma, total_concentration=1.0, scale_factor=scale_factor)
print("理想稳态浓度分布", {f'P{i}': c for i, c in enumerate(concentrations, start=1)})

k = {'k0': 1.0, 'k1': 1.0, 'k2': 2.0, 'k3': 1.6957874081758322, 'k4': 1.4450547072841455, 'k5': 1.2375667836122817, 'k6': 1.0651836020137946, 'k7': 0.9214075619979318, 'k8': 0.8010332521816378, 'k9': 0.6998754982223112, 'k10': 0.6145574772022625, 'k11': 0.5423450700912, 'k12': 0.48101692641668437, 'k13': 0.42876220285395594, 'k14': 0.3840998172415082, 'k15': 0.3458144845834327, 'k16': 0.3129058853499093, 'k17': 0.2845481431730273, 'k18': 0.26005742175685187, 'k19': 0.23886593653343924, 'k20': 0.2205010506089705, 'k21': 0.20456841343107499, 'k22': 0.19073832443109925, 'k23': 0.1787346778435064, 'k24': 0.16832598051462083, 'k25': 0.15931804057179608, 'k26': 0.151548008045691, 'k27': 0.14487951406850294, 'k28': 0.1391987070665381, 'k29': 0.1344110254794996, 'k30': 0.1304385793362551, 'k31': 0.12721803933754366, 'k32': 0.12469895337934692, 'k33': 0.12284242783000031, 'k34': 0.12162012525043595, 'k35': 0.12101354234516085, 'k36': 0.12101354234516085, 'k37': 0.12162012525043595, 'k38': 0.12284242783000031, 'k39': 0.12469895337934692, 'k40': 0.12721803933754366, 'k41': 0.1304385793362551, 'k42': 0.1344110254794996, 'k43': 0.13919870706653809, 'k44': 0.14487951406850294, 'k45': 0.15154800804569102, 'k46': 0.1593180405717961, 'k47': 0.16832598051462086, 'k48': 0.17873467784350644, 'k49': 0.19073832443109928, 'k50': 0.20456841343107504, 'k51': 0.22050105060897052, 'k52': 0.23886593653343927, 'k53': 0.2600574217568519, 'k54': 0.28454814317302735, 'k55': 0.31290588534990943, 'k56': 0.34581448458343284, 'k57': 0.3840998172415083, 'k58': 0.42876220285395605, 'k59': 0.48101692641668453, 'k60': 0.5423450700912001, 'k61': 0.6145574772022626, 'k62': 0.6998754982223113, 'k63': 0.801033252181638, 'k64': 0.9214075619979322, 'k65': 1.0651836020137948, 'k66': 1.2375667836122817, 'k67': 1.4450547072841455, 'k68': 1.6957874081758322, 'k69': 2.0}
k_inv = {'k1_inv': 1e-08, 'k2_inv': 0.09819745226687598, 'k3_inv': 0.09064767336107464, 'k4_inv': 0.08388780662701756, 'k5_inv': 0.07782636618060478, 'k6_inv': 0.07238363737162043, 'k7_inv': 0.06749005614761212, 'k8_inv': 0.0630848263471052, 'k9_inv': 0.05911473820268584, 'k10_inv': 0.0555331572724019, 'k11_inv': 0.052299157957104304, 'k12_inv': 0.049376779876079406, 'k13_inv': 0.046734388806477616, 'k14_inv': 0.04434412676092815, 'k15_inv': 0.0421814381787844, 'k16_inv': 0.04022466121908523, 'k17_inv': 0.038454674833035024, 'k18_inv': 0.03685459371468908, 'k19_inv': 0.035409504425322855, 'k20_inv': 0.03410623699673171, 'k21_inv': 0.03293316717220834, 'k22_inv': 0.03188004516674506, 'k23_inv': 0.030937847441392448, 'k24_inv': 0.03009864850843186, 'k25_inv': 0.029355510228930386, 'k26_inv': 0.028702386444778325, 'k27_inv': 0.028134041113909512, 'k28_inv': 0.02764597839887627, 'k29_inv': 0.027234383402741372, 'k30_inv': 0.026896072458686394, 'k31_inv': 0.026628452066228044, 'k32_inv': 0.026429485732137988, 'k33_inv': 0.02629766812212224, 'k34_inv': 0.026232006063589088, 'k35_inv': 0.026232006063589088, 'k36_inv': 0.02629766812212224, 'k37_inv': 0.026429485732137988, 'k38_inv': 0.026628452066228044, 'k39_inv': 0.026896072458686394, 'k40_inv': 0.027234383402741372, 'k41_inv': 0.02764597839887627, 'k42_inv': 0.028134041113909512, 'k43_inv': 0.028702386444778325, 'k44_inv': 0.029355510228930386, 'k45_inv': 0.03009864850843186, 'k46_inv': 0.030937847441392448, 'k47_inv': 0.03188004516674506, 'k48_inv': 0.03293316717220833, 'k49_inv': 0.03410623699673171, 'k50_inv': 0.035409504425322855, 'k51_inv': 0.03685459371468908, 'k52_inv': 0.038454674833035024, 'k53_inv': 0.04022466121908523, 'k54_inv': 0.0421814381787844, 'k55_inv': 0.04434412676092815, 'k56_inv': 0.046734388806477616, 'k57_inv': 0.049376779876079406, 'k58_inv': 0.052299157957104304, 'k59_inv': 0.0555331572724019, 'k60_inv': 0.05911473820268584, 'k61_inv': 0.0630848263471052, 'k62_inv': 0.06749005614761211, 'k63_inv': 0.07238363737162042, 'k64_inv': 0.07782636618060478, 'k65_inv': 0.08388780662701757, 'k66_inv': 0.09064767336107465, 'k67_inv': 0.098197452266876, 'k68_inv': 0.10664230289692239, 'k69_inv': 0.11610329519589954}
n = [2, 35, 69]
for i in range(3):
    k_optimized = list(k.values()) + list(k_inv.values())
    # print(k_optimized)
    k_optimized[n[i]] = k_optimized[n[i]] * 100
    k_optimized[n[i] + 69] = k_optimized[n[i] + 69] * 100

    initial_conditions = [5 + (concentrations[0] / 2.0), 5 - (concentrations[0] / 2.0)] + [0] * 70
    t = np.linspace(0, 10000, 5000)
    sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

    x_values = [f'P{i}' for i in range(1, 71)]
    # 绘制理想稳态浓度曲线
    plt.figure(figsize=(20, 10))
    plt.xlabel("P-Species")
    plt.ylabel("P-Concentrations")
    plt.title("Ideal Concentrations and Actual Concentrations")
    plt.xticks(range(len(x_values)), x_values, rotation=90)
    final_concentrations = sol[-1, 2:]
    plt.plot(range(len(x_values)), concentrations, label = 'Ideal Concentrations', marker='o', linestyle='-', color='blue')
    plt.plot(range(len(x_values)), final_concentrations, label = 'Actual Concentrations', marker='o', linestyle='-', color='red')
    plt.grid(True)
    plt.show()

    # 绘制各个物质的浓度变化曲线
    plt.figure(figsize=(20, 10))
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

    plt.figure(figsize=(20, 10))
    for i in range(12, 22):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P11-P20 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(22, 32):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P21-P30 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(32, 42):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P31-P40 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(42, 52):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P41-P50 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(52, 62):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P51-P60 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(62, 72):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P61-P70 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(3, 9):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P2-P7 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(31, 37):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P30-P35 Concentration over Time')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(20, 10))
    for i in range(66, 72):
        plt.plot(t, sol[:, i], label=f'p{i - 1}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Concentration')
    plt.title('P65-P70 Concentration over Time')
    plt.grid(True)
    plt.show()