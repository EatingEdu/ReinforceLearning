#####################  train.py  ####################

from ndqn import DQN
# import gym
import numpy as np
import time
import point


EPISODE = 100 # Episode limitation
STEP = 70 # Step limitation in an episode
TEST = 5 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = point.point_env()
  agent = DQN(env)
  print("===============================start train===================================")
  for episode in range(EPISODE):
    # initialize task
    state = 0
    # Train
    ep_reward = 0
    for step in range(STEP):
      action = agent.egreedy_action(np.array([state]))
      next_state,reward,done,_ = env.step(state,action)
    #   state = np.array([0])
       # e-greedy action for train
    #   print("=================================================")
    #   print(next_state)
    #   print("=================================================")
      env.update_env(next_state,episode,step)
    #   reward = 10 if done else -1
      ep_reward += reward
      agent.perceive(state,action,reward,next_state,done,episode)
      state = next_state
      if done:
        print('episode {} complete, ep_reward: {}'.format(episode,ep_reward))
        break
    # print('episode {} complete, reward: {}'.format(episode,reward))
    # Test every 10 episodes
    # agent.save('my_'+episode+'.model',)
    # agent.saver.restore(agent, "./save_restore_model/my_test")
    
    if episode % 20 == 0 :
      print("===============================start test===================================")
      total_reward = 0
      for i in range(TEST):
        state = np.array([0])
        for j in range(10):
          action = agent.action(state) # direct action for test
          next_state,reward,done,_ = env.step(state,action)
          print("test {} step {}, reward is :{}".format(i,j,reward))
          total_reward += reward
          if done:
            print("done")
            break
          else:
            state = next_state
          if type(next_state) != int and type(next_state) != type(np.int32(1)):
            next_state = next_state[0]
          env.update_env(next_state,episode,j)
          state = np.array([state])
        # print()
      ave_reward = total_reward/TEST
      print(total_reward)
      print ('episode: {} Evaluation Average Reward:{}'.format(episode,ave_reward))
      print("===============================end test===================================")
    # def save():

if __name__ == '__main__':
  main()