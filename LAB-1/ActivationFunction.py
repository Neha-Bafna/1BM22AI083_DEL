import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  y=1.0/(1+math.exp(-x))
  return y

def relu(x):
  return max(0,x)

def tanh(x):
  return np.tanh(x)

def linear(x):
  return x

def unit(x):
  return 1 if x>0 else 0

def sign(x):
  return 1 if x>0 else (-1 if x<0 else 0)

def activate(inp, weight):
  h=0
  for x,w in zip(inp, weight):
    h+=x*w
  return [sigmoid(h), relu(h), tanh(h), linear(h), unit(h), sign(h)]

if __name__=="__main__":
  inp=[.5, .3, .2]
  weight=[.4, .7, .2]
  output=activate(inp, weight)
  print(f"Sigmoid: {output[0]}")
  print(f"ReLU: {output[1]}")
  print(f"tanh: {output[2]}")
  print(f"linear: {output[3]}")
  print(f"unit: {output[4]}")
  print(f"sign: {output[5]}")

x_values = np.linspace(-10, 10, 400)
def plot_activation(activation, name):
    y_values = [activation(x) for x in x_values]
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=name,color='lightcoral')
    plt.title(f'{name} Activation Function')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()
    plt.show()
    print()

plot_activation(sigmoid, 'Sigmoid')
plot_activation(relu, 'ReLU')
plot_activation(tanh, 'Tanh')
plot_activation(linear, 'Linear')
plot_activation(unit, 'Unit Step')
plot_activation(sign, 'Sign')
