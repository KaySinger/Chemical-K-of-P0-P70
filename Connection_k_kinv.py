import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 数据：反应速率常数 k 和聚合物浓度 P
k = np.zeros(40)
k[0] = 1
for i in range(1, 40):
    k[i] = math.log(2**i)

print(k)

p = [0.062429655795787034, 0.07549303615293615, 0.09038157367707772, 0.10712971774940927, 0.12571788838521603, 0.14606334764943632, 0.16801284177452094,
     0.1913377929758532, 0.2157327591684209, 0.24081774813929327, 0.2661447718000128, 0.29120876504616566, 0.3154626891681067, 0.3383363150132504, 0.3592578643045527,
     0.37767740888915596, 0.39309071634089765, 0.40506211090851374, 0.4132449083429321, 0.41739808871846035, 0.41739808871846035, 0.4132449083429321,
     0.40506211090851374, 0.39309071634089765, 0.37767740888915596, 0.3592578643045527, 0.3383363150132504, 0.3154626891681067, 0.29120876504616566,
     0.2661447718000128, 0.24081774813929327, 0.2157327591684209, 0.1913377929758532, 0.16801284177452094, 0.14606334764943632, 0.12571788838521603,
     0.10712971774940927, 0.09038157367707772, 0.07549303615293615, 0.062429655795787034]

for i in range(40):
   p[i] = p[i] * 10

# 假设的模型：k = a * P^b
def model(P, a, x):
    return a * P**x

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(model, p[1: 39], k[2:], maxfev= 1000)

# 拟合得到的参数
a, x = popt
print(f"正向系数拟合参数: a = {a}, x = {x}")

# 使用拟合参数绘制拟合曲线
P_fit = np.linspace(min(p[1: 39]), max(p[1: 39]), 100)
k_fit = model(P_fit, *popt)

# 绘制原始数据和拟合曲线
plt.scatter(p[1: 39], k[2:], label='Natural data')
plt.plot(P_fit, k_fit, color='red', label=f'curve_fitting : k = {a:.2f} * P^{x:.2f}')
plt.xlabel('Concentration P')
plt.ylabel('k')
plt.legend()
plt.title('k vs P Curve_fitting')
plt.show()
#
# popt_inv, pcov_inv = curve_fit(model, p[2: 40], k_inv[1:])
#
# # 拟合得到的参数
# b, m = popt_inv
# print(f"逆向系数拟合参数: b = {b}, m = {m}")
#
# # 使用拟合参数绘制拟合曲线
# P_inv_fit = np.linspace(min(p[2: 40]), max(p[2: 40]), 100)
# k_inv_fit = model(P_inv_fit, *popt_inv)
#
# # 绘制原始数据和拟合曲线
# plt.scatter(p[2: 40], k_inv[1:], label='Natural data')
# plt.plot(P_inv_fit, k_inv_fit, color='red', label=f'curve_fitting : k_inv = {b:.2f} * P^{m:.2f}')
# plt.xlabel('Concentration P')
# plt.ylabel('k_inv')
# plt.legend()
# plt.title('k_inv vs P Curve_fitting')
# plt.show()