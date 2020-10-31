import time
FRESH_TIME = 0.1
class point_env():
    def __init__(self):
        self.N_STATES = 6

    #原来的get_env_feedback
    def step(self, state, action):
        if type(state) != int:
            state = state[0]
        # print("the current state is: {}, the current action is :{}".format(state,action))
        done = False
        if action == 1:
            if state == self.N_STATES - 2:
                next_state = 'terminal'
                R = 10
                done = True
            else:
                next_state = state + 1
                R = 1
        else:
            R = -1
            if state == 0:  
                next_state = state 
            else:
                next_state = state - 1
        # print('\nS: {}-{}-S\': {}'.format(state, action, next_state))
        return next_state, R, done, "_"


    # 环境更新
    # 每走完一步，要更新一帧图
    def update_env(self,state, episode, step_counter):
        env_list = ['-'] * (self.N_STATES - 1) + ['T']
        if state == 'terminal':
            interaction = 'episode: %s; total_steps = %s' % (episode, step_counter)  # fixme +1？？？
            # print('\r{}'.format(interaction))
            time.sleep(2)
            # print('\r                           ', end='')  # 清屏
        else:
            # state = state[0]
            # print(state)
            env_list[state] = 'o'
            # interaction = ''.join(env_list)
            # print('\r{}'.format(interaction),
            #     end='')  # end=''是Python3的内容，必须在文件导入的部分第一句位置写from __future__ import print_function
            # \r是回车，回到一行的开始
            time.sleep(FRESH_TIME)
        


