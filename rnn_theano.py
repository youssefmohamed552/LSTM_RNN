from nn_theano import NeuralNetwork as NN , sigmoid, tanh
from theano import printing, function,pp, shared
import theano.tensor as T
import random as rand
import numpy as np
import matplotlib.pyplot as plt



class System:
  def __init__(self):
    self.food = ['chicken', 'steak', 'pasta', 'fish']
    self.weather = ['rainy', 'surnny']
    self.mapped_food = self.map_array(self.food)
    self.encoded_food = self.encode(self.food)
    self.encoded_weather = self.encode(self.weather)


  def encode(self, arr):
    num_of_elements = len(arr)
    encoding = {}
    for i in range(num_of_elements):
      code = [[0] * num_of_elements]
      code[0][i] = 1.
      encoding[arr[i]] = code
    return encoding

  def map_array(self, arr):
    num_of_elements = len(arr)
    mapping = {}
    for i in range(num_of_elements):
      mapping[arr[i]] = i
    return mapping

  def get_index(self, element):
    return element.index(max(element))

  def num_of_vars(self):
    return (2 * len(self.food)) + len(self.weather)

  def get_example(self, prev=None):
    n = len(self.food)
    zeros = [[0] * n]
    encoded_prev = zeros
    if prev is None:
      current_index = 0
    else:
      current_index = self.mapped_food[prev]
      encoded_prev = self.encoded_food[prev]

    food = self.food[current_index]
    next_food = self.food[(current_index + 1) % n]
    weather = self.weather[rand.randrange(len(self.weather))]
    inp = [self.encoded_weather[weather][0] + self.encoded_food[food][0] + encoded_prev[0]]
    out = self.encoded_food[next_food] if weather is 'rainy' else self.encoded_food[food]
    # out = np.array(out).T.tolist()
    return inp, out



class RNN:
  def __init__(self, layers):
    self.nn = NN(tanh, layers)
    self.fnn = NN(sigmoid, layers)
    self.inn = NN(sigmoid, layers)
    self.snn = NN(sigmoid, layers)
    self.c_t_1 = T.matrix('c_t_1')
    self.c_t = T.matrix('c_t')
    self.memory = [[1] * layers[-1]]



    self.l = len(layers)-1

    # variables
    self.x = T.matrix('x')
    self.nn.eval(self.x)
    self.fnn.eval(self.x)
    self.snn.eval(self.x)
    self.inn.eval(self.x)
    self.a_hat = T.matrix('a_hat')
    self.ws = self.nn.ws + self.fnn.ws + self.snn.ws + self.inn.ws
    self.b = self.nn.b + self.fnn.b + self.snn.b + self.inn.b

    self.c_t = (self.fnn.a + self.c_t_1) + (self.nn.a * self.inn.a)
    self.a =  tanh(self.c_t) * self.snn.a


    self.alpha = 0.1

    # back prob
    cost = -(self.a_hat*T.log(self.a) + (1-self.a_hat)*T.log(1-self.a)).sum()
    parameters = self.ws + self.b
    grad = T.grad(cost, parameters)



    # create functions
    list_of_weight_updates = [
      [self.ws[i], self.ws[i]-self.alpha*grad[i]] for i in range(4*self.l)
    ]
    list_of_bias_updates = [
      [self.b[i], self.b[i]-self.alpha*grad[i+(4*self.l)]] for i in range(4*self.l)
    ]
    list_of_updates = list_of_weight_updates + list_of_bias_updates
    self.train_rnn = function(
      inputs = [self.c_t_1, self.x, self.a_hat],
      outputs = [self.c_t, self.a, cost],
      updates = list_of_updates
    )


    self.test_rnn = function(
      inputs = [self.c_t_1, self.x, self.a_hat],
      outputs = [self.c_t, self.a, cost],
    )




  def train(self, inp, y):
    c_t, pred, cost = self.train_rnn(self.memory, inp, y)
    self.memory = c_t
    return pred, cost
    

  def test(self, inp, y):
    c_t, pred, cost = self.test_rnn(self.memory, inp, y)
    self.memory = c_t
    return pred, cost
    

  def drop_out(self, p):
    self.nn.drop_out(p)


class Trainer:
  def __init__(self):
    self.s = System()
    inp_count = len(self.s.food) * 2 + len(self.s.weather)
    out_count = len(self.s.food)
    self.rnn = RNN([inp_count, out_count])
    self.errors = []
    self.errors_test = []

  def train(self, sample_size, learning_rate):
    self.rnn.alpha = learning_rate
    inp, y = self.s.get_example()
    for _ in range(sample_size):
      res, cost = self.rnn.train(inp, y)
      inp, y = self.s.get_example(self.s.food[self.s.get_index(res.tolist())])
      self.errors.append(cost)

  def display(self):
    plt.plot(self.errors)
    plt.show()

  def test(self, sample_size):
    inp, y = self.s.get_example()
    for _ in range(sample_size):
      res, cost = self.rnn.test(inp, y)
      inp, y = self.s.get_example(self.s.food[self.s.get_index(res.tolist())])
      err = 1.0  if self.s.get_index(y) == self.s.get_index(res.tolist()) else 0.0
      self.errors_test.append(err)
    accuracy = ((sum(self.errors_test)/len(self.errors_test)) * 100.0)
    print("accuracy {}%".format(accuracy))

  def get_new_skill(self, p):
    self.rnn.drop_out(p)
    
    




if __name__ == '__main__':
  trainer = Trainer()
  for _ in range(10):
    trainer.train(1000, 1)
    trainer.get_new_skill(0.1)

  trainer.train(1000, 1)
  trainer.display()
  trainer.test(1000)

