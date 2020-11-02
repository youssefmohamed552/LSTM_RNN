from nn_theano import NeuralNetwork as NN , sigmoid, tanh
from theano import printing, function,pp, shared
import theano.tensor as T
import random as rand
import numpy as np
import matplotlib.pyplot as plt



class System:
  def __init__(self):
    # self.food = ['chicken', 'steak', 'pasta', 'fish']
    self.food = ['chicken', 'steak']
    self.weather = ['rainy', 'surnny']
    self.mapped_food = self.map_array(self.food)
    self.encoded_food = self.encode(self.food)
    self.encoded_weather = self.encode(self.weather)


  def encode(self, arr):
    num_of_elements = len(arr)
    encoding = {}
    for i in range(num_of_elements):
      # code = np.zeros((num_of_elements,1))
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
    m = element[0][0]
    ind = 0
    for i in range(1, len(element[0])): 
      if element[0][i] > m:
        m = element[0][i]
        ind = i
    return ind


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

    self.c_t = (self.c_t_1 * self.snn.a) + (self.nn.a * self.fnn.a)
    self.a =  tanh(self.c_t) * self.inn.a


    self.alpha = 0.1

    # back prob
    cost = -(self.a_hat*T.log(self.a) + (1-self.a_hat)*T.log(1-self.a)).sum()
    parameters = self.ws + self.b
    grad = T.grad(cost, parameters)
    # self.backprob(self.nn.a)



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
    # pred_nn, cost_nn = self.nn.train(self.memory,inp, inp, inp, inp, y)
    c_t, pred_nn, cost_nn = self.train_rnn(self.memory, inp, y)
    self.memory = c_t
    # print(c_t)
    # print(pred_nn)
    # print(cost_nn)
    # pred_fnn, cost_fnn = self.fnn.train(self.memory,inp, inp, inp, inp, y)
    # pred_snn, cost_snn = self.snn.train(self.memory, inp, inp, inp, inp, y)
    # pred_inn, cost_inn = self.inn.train(self.memory, inp, inp, inp, inp, y)
    # m1 = [a*b for a,b in zip(self.memory,pred_snn)]
    # m2 = [a*b for a,b in zip(pred_fnn, pred_nn)]
    # a = [a+b for a,b in zip(m1,m2)]
    # self.memory = a
    # t = np.tanh(np.array(a)).tolist()
    # m3 = [a*b for a,b in zip(pred_inn, t)]
    # cost = (cost_nn + cost_fnn + cost_snn + cost_inn) /4

    # return m3, cost
    return pred_nn, cost_nn
    

  def test(self, inp, y):
    c_t, pred_nn, cost_nn = self.test_rnn(self.memory, inp, y)
    self.memory = c_t
    return pred_nn, cost_nn
    # print(c_t)
    

  def backprob(self, a):
    nn_cost, nn_grad = self.nn.backprob(a)
    self.nn.training_functions(nn_cost, nn_grad, [self.c_t_1, self.x, self.nn.a_hat], [self.c_t, self.nn.a, nn_cost])
    # fnn_cost, fnn_grad = self.fnn.backprob(a)
    # self.fnn.training_functions(fnn_cost, fnn_grad, [self.c, self.x, self.fnn.a_hat])
    # snn_cost, snn_grad = self.snn.backprob(a)
    # self.snn.training_functions(snn_cost, snn_grad, [self.c, self.x, self.snn.a_hat])
    # inn_cost, inn_grad = self.inn.backprob(a)
    # self.inn.training_functions(inn_cost, inn_grad, [self.c, self.x, self.inn.a_hat])



  def drop_out(self, p):
    self.nn.drop_out(p)




class Trainer:
  def __init__(self):
    self.s = System()
    inp_count = len(self.s.food) * 2 + len(self.s.weather)
    out_count = len(self.s.food)
    self.rnn = RNN([inp_count, 10 , 10 , out_count])
    self.errors = []
    self.errors_test = []

  def train(self, sample_size, learning_rate):
    self.rnn.alpha = learning_rate
    inp, y = self.s.get_example()
    for _ in range(sample_size):
      res, cost = self.rnn.train(inp, y)
      # self.rnn.train(inp, y)
      inp, y = self.s.get_example(self.s.food[self.s.get_index(res)])
      # print(cost)
      self.errors.append(cost)

    print(sum(self.errors[-100:])/len(self.errors[-100:]))
    plt.plot(self.errors)
    plt.show()

  def test(self, sample_size):
    inp, y = self.s.get_example()
    for _ in range(sample_size):
      res, cost = self.rnn.test(inp, y)
      inp, y = self.s.get_example(self.s.food[self.s.get_index(res)])
      err = 1.0 if self.s.get_index(y) == self.s.get_index(res) else 0.0
      self.errors_test.append(err)
    accuracy = ((sum(self.errors_test)/len(self.errors_test)) * 100.0)
    print("accuracy {}%".format(accuracy))




if __name__ == '__main__':
  trainer = Trainer()
  trainer.train(10000, 0.003)
  # trainer.get_new_skill(0.1)
  trainer.test(100)

