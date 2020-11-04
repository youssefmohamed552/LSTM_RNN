import numpy as np
import theano.tensor as T
from theano.compile.debugmode import DebugMode
from theano import function, shared, pp, config
from random import random

import matplotlib.pyplot as plt
config.exception_verbosity='high'
config.optimizer='None'

def sigmoid(z):
  return 1/(1+T.exp(-z))

def tanh(z):
  return T.tanh(z)



class NeuralNetwork:  
  def __init__(self, func, layers):
    self.l = len(layers)-1
    self.x = T.matrix('x')
    self.ws = [shared(np.random.rand(layers[i],layers[i+1])) for i in range(self.l)]
    self.b = [shared(1.) for _ in range(self.l)]
    self.a_hat = T.matrix('a_hat')

    self.alpha = 0.1

    self.func = func

    self.eval(self.x)
    cost, grad = self.backprob(self.a)
    self.training_functions(cost, grad, [self.x, self.a_hat], [self.a, cost])

  def training_functions(self, cost, grad, input_list, output_list):
    list_of_weight_updates = [
      [self.ws[i], self.ws[i]-self.alpha*grad[i]] for i in range(self.l)
    ]
    list_of_bias_updates = [
      [self.b[i], self.b[i]-self.alpha*grad[i+self.l]] for i in range(self.l)
    ]
    list_of_updates = list_of_weight_updates + list_of_bias_updates
    self.train = function(
      inputs = input_list,
      outputs = output_list,
      updates = list_of_updates,
    )


  def eval(self, x):
    self.a = x
    for i in range(self.l):
      z = T.dot(self.a, self.ws[i]) + self.b[i]
      self.a = self.func(z)

  def backprob(self, a):
    cost = -(self.a_hat*T.log(a) + (1-self.a_hat)*T.log(1-a)).sum()
    parameters = self.ws + self.b
    grad = T.grad(cost, parameters)
    return cost, grad

  def drop_out(self, probability):
    for w in self.ws:
      weights = w.get_value()
      f = np.random.binomial([np.ones(weights.shape)], 1-probability)[0]
      new_weight = np.multiply(weights, f)
      w.set_value(new_weight)


    # self.test = function(
      # inputs = [self.x, self.a_hat],
      # outputs = [self.a, cost]
    # )


    
    

class Trainer:
  def __init__(self, system, model):
    self.cost = []
    self.cost_test = []
    self.system = system
    self.model = model

  def start(self):
    for _ in range(10000):
      pred, cost_iter = self.model.train(self.system.inputs, self.system.outputs)
      self.cost.append(cost_iter)

  def test(self):
    for _ in range(1000):
      pred, cost_iter= self.model.test(self.system.inputs, self.system.outputs)
      self.cost_test.append(cost_iter)

    accuracy = 100.0 - ((sum(self.cost_test)/len(self.cost_test)) * 100)
    print("accuracy {}%".format(accuracy))
    

  def display(self):
    plt.plot(self.cost)
    plt.show()


class AndGate:
  def __init__(self):
    self.inputs = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1]
    ]

    self.outputs = [[0],[0],[0],[1]]




if __name__ == '__main__':
  t = Trainer(AndGate(), NeuralNetwork(sigmoid, [2, 4, 4, 1]))
  t.start()
  # t.test()
  t.display()




