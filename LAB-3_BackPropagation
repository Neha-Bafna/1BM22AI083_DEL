import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Use 1D array for binary output
y = np.array([0, 1, 1, 0])  

input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 2
np.random.seed(42)

weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
bias_hidden = np.zeros((1, hidden_layer_size))

weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size)
bias_output = np.zeros((1, output_layer_size))

def feedforward(X):
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = relu(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    output = softmax(output_input)
    return hidden_output, output

def backpropagation(X, y, learning_rate=0.01):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    hidden_output, output = feedforward(X)

    # Convert y to one-hot encoding for output
    y_one_hot = np.eye(output_layer_size)[y]

    # Calculate the output error using MSE
    output_error = output - y_one_hot
    output_delta = output_error 
    hidden_error = output_delta.dot(weights_hidden_output.T) * relu_derivative(hidden_output)
    hidden_delta = hidden_error

    weights_hidden_output -= hidden_output.T.dot(output_delta) * learning_rate
    bias_output -= np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    weights_input_hidden -= X.T.dot(hidden_delta) * learning_rate
    bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

for _ in range(10000):
  backpropagation(X, y, learning_rate=0.001)

hidden_output, final_output = feedforward(X)

print("Output probabilities for each input:")
print(final_output)
predictions = np.argmax(final_output, axis=1)
print("\nPredictions:", predictions)
