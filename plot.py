import numpy as np
import matplotlib.pyplot as plt

def legendre_basis(n, eps):
    p = [1.0] * (n + 1)
    if n >= 1:
        p[1] = eps
    for i in range(1, n):
        p[i + 1] = ((2.0 * i + 1.0) * eps * p[i] - i * p[i - 1]) / (i + 1.0)
    return p

files = ["data_2.txt", "data_4.txt", "data_5.txt"]
degrees = [1, 2, 4, 8]
colors = ["blue", "green", "orange", "red"]

for filename in files:
    try:
        data = np.loadtxt(filename)
    except OSError:
        continue

    eps = data[:, 0]
    sigma = data[:, 1]

    sort_idx = np.argsort(eps)
    eps = eps[sort_idx]
    sigma = sigma[sort_idx]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(eps, sigma, color="black", s=10, alpha=0.3, label="Данные")

    for n, col in zip(degrees, colors):
        A = np.array([legendre_basis(n, e) for e in eps])
        c, _, _, _ = np.linalg.lstsq(A, sigma, rcond=None)
        f_pred = A @ c
        plt.plot(eps, f_pred, label=f"N = {n}", color=col, linewidth=2)

    plt.title(f"Аппроксимация для {filename}", fontsize=14)
    plt.xlabel(r"$\varepsilon$", fontsize=12)
    plt.ylabel(r"$\sigma$", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_name = filename.replace(".txt", ".png")
    plt.savefig(out_name)
    plt.show()