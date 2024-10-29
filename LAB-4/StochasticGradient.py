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
num_samples = X.shape[0]
losses = []

for epoch in range(num_epochs):
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for i in indices:
        z = X[i].dot(weights) + bias
        y_pred = softmax(z.reshape(1, -1))
        loss = log_loss(y[i:i+1], y_pred)

        grad_weights = X[i].reshape(-1, 1).dot((y_pred - np.eye(num_classes)[y[i]]).reshape(1, -1))
        grad_bias = y_pred - np.eye(num_classes)[y[i]].reshape(1, -1)

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

    epoch_loss = log_loss(y, softmax(X.dot(weights) + bias))
    losses.append(epoch_loss)  
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')

final_z = X.dot(weights) + bias
final_pred = softmax(final_z)
final_loss = log_loss(y, final_pred)
print(f'Final Log Loss: {final_loss:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss', color='blue')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
