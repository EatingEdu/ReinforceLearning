# import gym
import ddqn
import point
from ddqn import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# env = gym.make('Pendulum-v0')
# env = env.unwrapped
# env.seed(1)
BATCH_SIZE = 32
MEMORY_SIZE = 100
ACTION_SPACE = 2
EPISODE = 100 # Episode limitation
STEP = 30 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=1, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=1, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    env = point.point_env()
    # observation = env.reset()
    for episode in range(EPISODE):
        # observation = 0
        observation = np.array([0])
        # print(observation)
        ep_reward = 0
        for step in range(STEP):
            
            # if total_steps - MEMORY_SIZE > 8000: env.render()
            # print(observation)
            action = RL.choose_action(observation)

            # f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(observation,action)

            ep_reward += reward    # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
            # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

            RL.store_transition(observation, action, reward, observation_)

            if step > BATCH_SIZE:   # learning
                RL.learn()
            if done:
                print('episode {} complete, ep_reward: {}'.format(episode,ep_reward))
                break
            
            observation = np.array([observation_])
            total_steps += 1
    return RL.q

# def test():
    
q_natural = train(natural_DQN)
q_double = train(double_DQN)

print(len(q_natural),len(q_double))
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()