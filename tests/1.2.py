import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# 1. Загрузка данных
iris = load_iris()
X = iris.data[:, :2]          # длина и ширина чашелистика
y = iris.target

# 2. Берём только 2 класса: setosa(0) и versicolor(1)
mask = y < 2
X, y = X[mask], y[mask]

# 3. Обучение персептрона
model = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
model.fit(X, y)

# 4. Результаты
y_pred = model.predict(X)

print("Веса:", model.coef_)
print("Смещение:", model.intercept_)
print("Точность:", accuracy_score(y, y_pred))