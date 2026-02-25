import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100
x = np.random.randn(n_samples) * 2 
true_w, true_b = 3, 1
epsilon = np.random.randn(n_samples) * 2  
y = true_w * x + true_b + epsilon

class SGDWithMiniBatches:
    def __init__(self, learning_rate=0.02, batch_size=10):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.w = 10.0
        self.b = 10.0
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        
    def compute_loss(self, x, y):
        y_pred = self.w * x + self.b
        loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def compute_gradients(self, x_batch, y_batch):
        y_pred = self.w * x_batch + self.b
        error = y_batch - y_pred
        dw = -2 * np.mean(error * x_batch)
        db = -2 * np.mean(error)
        return dw, db
    
    def train(self, x, y, epochs=50):
        n_samples = len(x)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            for i in range(0, n_samples, self.batch_size):
                x_batch = x_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                dw, db = self.compute_gradients(x_batch, y_batch)
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            loss = self.compute_loss(x, y)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
        return self.w, self.b

class RegularSGD:
    def __init__(self, learning_rate=0.02):
        self.lr = learning_rate
        self.w = 10.0
        self.b = 10.0
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        
    def compute_loss(self, x, y):
        y_pred = self.w * x + self.b
        loss = np.mean((y - y_pred) ** 2)
        return loss
    
    def train(self, x, y, epochs=50):
        n_samples = len(x)
        
        for epoch in range(epochs):
            # Перемешиваем данные
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]
            
            # Проходим по каждому примеру
            for i in range(n_samples):
                x_i = x_shuffled[i:i+1]
                y_i = y_shuffled[i:i+1]
                
                # Вычисляем градиенты
                y_pred = self.w * x_i + self.b
                error = y_i - y_pred
                dw = -2 * error * x_i
                db = -2 * error
                
                # Обновляем параметры
                self.w -= self.lr * dw[0]
                self.b -= self.lr * db[0]
            
            # Сохраняем историю
            loss = self.compute_loss(x, y)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            
        return self.w, self.b


sgd_mini = SGDWithMiniBatches(learning_rate=0.02, batch_size=10)
w_mini, b_mini = sgd_mini.train(x, y, epochs=50)
loss_mini = sgd_mini.compute_loss(x, y)

sgd_regular = RegularSGD(learning_rate=0.02)
w_reg, b_reg = sgd_regular.train(x, y, epochs=50)
loss_reg = sgd_regular.compute_loss(x, y)

print("=" * 50)
print("SGD с мини-батчами (batch_size=10):")
print(f"Финальные параметры: w = {w_mini:.4f}, b = {b_mini:.4f}")
print(f"Финальный loss: {loss_mini:.6f}")
print(f"Истинные параметры: w = {true_w}, b = {true_b}")
print()
print("Обычный SGD:")
print(f"Финальные параметры: w = {w_reg:.4f}, b = {b_reg:.4f}")
print(f"Финальный loss: {loss_reg:.6f}")
print(f"Истинные параметры: w = {true_w}, b = {true_b}")
print("=" * 50)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


axes[0, 0].plot(sgd_mini.loss_history, 'b-', label='SGD с мини-батчами', linewidth=2)
axes[0, 0].plot(sgd_regular.loss_history, 'r--', label='Обычный SGD', linewidth=2)
axes[0, 0].set_xlabel('Эпоха')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Сходимость Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(sgd_mini.w_history, 'b-', label='SGD с мини-батчами', linewidth=2)
axes[0, 1].plot(sgd_regular.w_history, 'r--', label='Обычный SGD', linewidth=2)
axes[0, 1].axhline(y=true_w, color='g', linestyle=':', label=f'Истинное w={true_w}')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('w')
axes[0, 1].set_title('Сходимость параметра w')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(sgd_mini.b_history, 'b-', label='SGD с мини-батчами', linewidth=2)
axes[1, 0].plot(sgd_regular.b_history, 'r--', label='Обычный SGD', linewidth=2)
axes[1, 0].axhline(y=true_b, color='g', linestyle=':', label=f'Истинное b={true_b}')
axes[1, 0].set_xlabel('Эпоха')
axes[1, 0].set_ylabel('b')
axes[1, 0].set_title('Сходимость параметра b')
axes[1, 0].legend()
axes[1, 0].grid(True)


axes[1, 1].scatter(x, y, alpha=0.6, label='Данные')
x_line = np.linspace(min(x), max(x), 100)
axes[1, 1].plot(x_line, true_w * x_line + true_b, 'g-', label='Истинная линия', linewidth=2)
axes[1, 1].plot(x_line, w_mini * x_line + b_mini, 'b--', label='SGD с мини-батчами', linewidth=2)
axes[1, 1].plot(x_line, w_reg * x_line + b_reg, 'r:', label='Обычный SGD', linewidth=2)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('y')
axes[1, 1].set_title('Данные и линии регрессии')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(sgd_mini.loss_history, 'b-', label='SGD с мини-батчами', linewidth=2)
plt.plot(sgd_regular.loss_history, 'r--', label='Обычный SGD', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Loss (MSE)')
plt.title('Сравнение траекторий сходимости')
plt.legend()
plt.grid(True)
plt.show()