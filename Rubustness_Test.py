import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def equations(p, t, k_values):
    k = k_values[:70]
    k_inv = k_values[70:]
    dpdt = [0] * 71
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1]**2
    for i in range(2, 70):
        dpdt[i] = k[i-1] * p[i-1]**2 + k_inv[i-1] * p[i+1] - k_inv[i-2] * p[i] - k[i] * p[i]**2
    dpdt[70] = k[69] * p[69]**2 - k_inv[68] * p[70]

    return dpdt

k = [1, 2.00006486, 1.68737886, 1.43071306, 1.21917092, 1.04411588,
 0.89867811, 0.77737608, 0.67581775, 0.5904722,  0.51849047, 0.45756574,
 0.4058238,  0.36173697, 0.32405621, 0.29175567, 0.26399137, 0.24006652,
 0.21940419, 0.20152537, 0.1860313,  0.17258925, 0.16092111, 0.15079388,
 0.14201229, 0.13441247, 0.12785706, 0.12223099, 0.1174382,  0.11339893,
 0.11004745, 0.10733034, 0.10520502, 0.10363869, 0.10260744, 0.10209567,
 0.10209563, 0.10260741, 0.10363865, 0.10520498, 0.10733028, 0.1100474,
 0.11339888, 0.11743815, 0.12223093, 0.12785701, 0.13441242, 0.14201222,
 0.15079382, 0.16092103, 0.17258919, 0.1860312,  0.20152528, 0.21940406,
 0.2400664,  0.2639912,  0.29175553, 0.32405606, 0.3617367,  0.40582283,
 0.45756473, 0.51848976, 0.59047143, 0.67581704, 0.77737531, 0.89867749,
 1.04411536, 1.21917091, 1.43071467, 1.68738239]
k_inv = [0.89973278, 0.82848152, 0.76478225, 0.7077488,  0.65660897, 0.61068899,
 0.56940227, 0.53223565, 0.49874027, 0.4685227,  0.44123758, 0.41658145,
 0.39428747, 0.37412123, 0.35587502, 0.33936605, 0.32443297, 0.31093334,
 0.29874136, 0.28774591, 0.27784889, 0.26896389, 0.26101472, 0.25393454,
 0.24766478, 0.24215447, 0.23735942, 0.23324168, 0.22976908, 0.22691478,
 0.22465687, 0.22297818, 0.221866,   0.22131197, 0.22131194, 0.22186583,
 0.22297799, 0.22465665, 0.22691455, 0.22976884, 0.23324143, 0.23735915,
 0.2421542,  0.2476645,  0.25393426, 0.26101444, 0.26896359, 0.2778486,
 0.28774557, 0.29874102, 0.31093295, 0.32443258, 0.33936559, 0.35587461,
 0.37412078, 0.39428702, 0.41658038, 0.44123583, 0.46852038, 0.49873817,
 0.53223397, 0.56940056, 0.61068745, 0.65660733, 0.70774732, 0.76477996,
 0.82847713, 0.8997278,  0.97955327]
k_optimized = k + k_inv

initial_conditions = [100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

t = np.linspace(0, 500, 1000)

sol = odeint(equations, initial_conditions, t, args=(k_optimized,))

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(0, 11):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P0-P10 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(11, 21):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P11-P20 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(21, 31):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P21-P30 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(31, 41):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P31-P40 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(41, 51):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P41-P50 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(51, 61):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P51-P60 Concentration over Time')
plt.grid(True)
plt.show()

# 各个物质的浓度变化曲线图
plt.figure(figsize = (10, 6))
for i in range(61, 71):
    plt.plot(t, sol[:, i], label = f'p{i}')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('P61-P70 Concentration over Time')
plt.grid(True)
plt.show()

# 绘制动态平衡时各个物质的浓度曲线图
plt.figure(figsize=(10, 6))
final_concentrations = sol[-1, 1:]
labels = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'p31', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p49', 'p50', 'p51', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p58', 'p59', 'p60', 'p61', 'p62', 'p63', 'p64', 'p65', 'p66', 'p67', 'p68', 'p69', 'p70']
plt.plot(labels, final_concentrations, 'o-', label = 'Simulated')
plt.xlabel('Species')
plt.ylabel('Concentration at Equilibrium')
plt.title('Concentrations at Equilibrium')
plt.grid(True)
plt.show()