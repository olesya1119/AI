import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# 1) Данные: y = 3x + 1 + eps
n = 100
x = np.random.uniform(-5, 5, n)
eps = np.random.normal(0, 1, n)
y = 3 * x + 1 + eps

lr = 0.02
epochs = 50

def mse(w, b):
    return np.mean((w * x + b - y) ** 2)

def train(batch_size):
    w, b = 10.0, 10.0
    losses = []

    for _ in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            batch = idx[i:i + batch_size]
            xb, yb = x[batch], y[batch]

            # предсказание и ошибка
            err = (w * xb + b) - yb

            # градиенты MSE
            dw = 2 * np.mean(err * xb)
            db = 2 * np.mean(err)

            # шаг градиентного спуска
            w -= lr * dw
            b -= lr * db

            # для сравнения считаем loss на ВСЕХ данных
            losses.append(mse(w, b))

    return w, b, losses

# 2) Обучение
w_sgd, b_sgd, loss_sgd = train(batch_size=1)
w_mb,  b_mb,  loss_mb  = train(batch_size=10)

# 3) Вывод
print(f"SGD (batch=1):  w = {w_sgd:.4f}, b = {b_sgd:.4f}, loss = {loss_sgd[-1]:.6f}")
print(f"Mini-batch=10: w = {w_mb:.4f},  b = {b_mb:.4f},  loss = {loss_mb[-1]:.6f}")

# 4) График сходимости
plt.plot(loss_sgd, label="SGD (batch=1)")
plt.plot(loss_mb,  label="Mini-batch (10)")
plt.xlabel("Шаг обновления")
plt.ylabel("MSE loss (на всех данных)")
plt.legend()
plt.grid(True)
plt.show()