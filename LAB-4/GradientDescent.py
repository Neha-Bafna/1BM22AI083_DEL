#Gradient Descent
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

num_classes = len(np.unique(y))
num_features = X.shape[1]
weights = np.random.randn(num_features, num_classes)
bias = np.zeros((1, num_classes))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

def log_loss(y_true, y_pred):
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(len(y_true)), y_true] = 1
    return -np.mean(np.sum(y_true_one_hot * np.log(y_pred + 1e-15), axis=1))

learning_rate = 0.01
num_epochs = 1000
loss_values=[]

for epoch in range(num_epochs):
    z = X.dot(weights) + bias
    y_pred = softmax(z)
    loss = log_loss(y, y_pred)
    loss_values.append(loss)
    grad_weights = X.T.dot(y_pred - np.eye(num_classes)[y]) / X.shape[0]
    grad_bias = np.mean(y_pred - np.eye(num_classes)[y], axis=0, keepdims=True)
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
        
final_z = X.dot(weights) + bias
final_pred = softmax(final_z)
final_loss = log_loss(y, final_pred)
print(f'Final Log Loss: {final_loss:.4f}')

plt.plot(range(num_epochs),loss_values)
plt.title("Loss function Over Epochs")
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.grid()
plt.show()


output:
Epoch 0, Loss: 2.5628
Epoch 100, Loss: 0.9861
Epoch 200, Loss: 0.5847
Epoch 300, Loss: 0.4729
Epoch 400, Loss: 0.4251
Epoch 500, Loss: 0.3972
Epoch 600, Loss: 0.3775
Epoch 700, Loss: 0.3620
Epoch 800, Loss: 0.3491
Epoch 900, Loss: 0.3379
Final Log Loss: 0.3278


