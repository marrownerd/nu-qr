import numpy as np
import matplotlib.pyplot as plt

def legendre_basis(n, eps):
    p = [1.0] * (n + 1)
    if n >= 1:
        p[1] = eps
    for i in range(1, n):
        p[i + 1] = ((2.0 * i + 1.0) * eps * p[i] - i * p[i - 1]) / (i + 1.0)
    return p

data = np.loadtxt("data_2.txt")
eps = data[:, 0]
sigma = data[:, 1]

sort_idx = np.argsort(eps)
eps = eps[sort_idx]
sigma = sigma[sort_idx]


plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(eps, sigma, color="black", s=10, alpha=0.3, label="Экспериментальные данные")


degrees = [1, 2, 4, 8]
colors = ["blue", "green", "orange", "red"]

for n, col in zip(degrees, colors):
    A = np.array([legendre_basis(n, e) for e in eps])
    
    c, _, _, _ = np.linalg.lstsq(A, sigma, rcond=None)

    f_pred = A @ c
    
    plt.plot(eps, f_pred, label=f"Полином Лежандра N = {n}", color=col, linewidth=2)

plt.title("Аппроксимация экспериментальных данных полиномами Лежандра", fontsize=14)
plt.xlabel(r"Деформация $\varepsilon$", fontsize=12)
plt.ylabel(r"Напряжение $\sigma$", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("plot_result.png")
print("1")
plt.show()