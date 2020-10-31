# DQN

import tensorflow as tf
import numpy as np
# import gym
import time
import random
from collections import deque

#####################  hyper parameters  ####################

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 1000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

###############################  DQN  ####################################

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    # self.N_state = 6 
    self.state_dim = 1
    self.actions = [0,1]
    self.action_dim = 2

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())
    # self.saver =  tf.train.Saver()
    

  #创建q_net网络，正向传播过程
  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,3]) #隐层中有3个神经元
    b1 = self.bias_variable([3])
    W2 = self.weight_variable([3,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2


#定义目标函数为两个网络的差值,在初始化的时候已经拿到了
  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

#初始化状态和动作值
  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    # one_hot_state= np.zeros(self.state_dim)
    # one_hot_state[state] = 1
    if next_state == "terminal":
        next_state = np.array([5])
    self.replay_buffer.append((np.array([state]),one_hot_action,reward,np.array([next_state]),done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE: #大于batch_size之后才可以进行训练
      self.train_Q_network()


  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]
    # print(minibatch)
    # Step 2: calculate y
    y_batch = []
    # print("next_state_batch is {}".format(next_state_batch))
    #这里的q_value_batch获取到的是在target_net中的所有的值,原始网络中的预测值;
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch}) #对于输入的所有state进行预测，获得q
    
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))  #从更新后的网络中来获取最大值吗?还是在之前的网络中获取最大值

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

#e-贪心算法,训练阶段使用
  def egreedy_action(self,state):
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(self.state_input,state)
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    if random.random() <= self.epsilon:
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.random.choice(self.actions)
    else:
        # self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.argmax(Q_value)

#贪心算法,在测试阶段使用
  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

#q-learning中的贪心算法
#   def egreedy_action(self,state):
#     state_actions = q_table.iloc[state, :]
#     if np.random.uniform() > self.epsilon:
#         action_name = np.random.choice(self.actions)
#     else:
#         action_name = state_actions.idxmax()#找出q_table中的q_值最大的action
#     return action_name


#初始化网络权重参数
  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

