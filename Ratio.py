import Rubustness_Test

k = Rubustness_Test.k
k_inv = Rubustness_Test.k_inv

Ratio = [0] * 69

for i in range(0, 69):
    Ratio[i] = k[i+1] / k_inv[i]

k = {f'k{i}': c for i, c in enumerate(k, start=0)}
k_inv = {f'k_inv{i}': c for i, c in enumerate(k_inv, start=1)}
Ratio = {f'Ratio{i}': c for i, c in enumerate(Ratio, start=1)}
print(k)
print(k_inv)
print("比值为",Ratio)