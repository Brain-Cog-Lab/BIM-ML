import numpy as np
import matplotlib.pyplot as plt

e_s = 40  # 到第 e_s 个 epoch 完全衰减到 0
b_l = 1
b_i = 0

e_c_values = np.linspace(0, e_s, 200)
P_values = []

for e_c in e_c_values:
    ratio = (b_i + e_c * b_l) / (e_s * b_l)
    ratio = np.clip(ratio, 0, 1)

    # 关键：使用 (1 - ratio)^3，得到开口朝上、从1降到0的曲线
    P_replacement = (1.0 - ratio) ** 3
    P_replacement = np.clip(P_replacement, 0, 1)
    P_values.append(P_replacement)

plt.figure(figsize=(6, 4))
plt.plot(e_c_values, P_values, label=r"$P_{\mathrm{replacement}} = (1 - \mathrm{ratio})^3$")
plt.title("Concave-Up Decaying Curve from 1.0 to 0.0")
plt.xlabel("epoch (e_c)")
plt.ylabel("P_replacement")
plt.ylim([-0.05, 1.05])
plt.grid(True)
plt.legend()
plt.show()
