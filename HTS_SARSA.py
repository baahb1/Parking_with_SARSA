import numpy as np
from tqdm import tqdm
import Enviroment
import rl_glue
class SARSA:
    def __init__(self,enviroment,step_size,epsilon,rows,columns,discount):
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_actions = 6
        self.discount = discount
        self.num_states = (rows * columns * 4)
        self.enviroment = enviroment

        self.rand_generator = np.random.RandomState(0)

        self.q = np.zeros(shape=(self.num_states,self.num_actions))


    def agent_start(self,observation):
        state = observation
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:

            #modify for Normal SARSA
            action = self.argmax(current_q)

        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_step(self,reward,observation):
        state = observation
        current_q = self.q[state,:]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        
        # Perform an update
        # --------------------------
        # your code here

        self.q[self.prev_state, self.prev_action] += self.step_size * (reward + (self.discount * np.average(current_q)) - self.q[self.prev_state, self.prev_action])
        # --------------------------

        self.prev_state = state
        self.prev_action = action
        return action
    
    def agent_end(self, reward):
        self.q[self.prev_state, self.prev_action] += self.step_size * (reward - self.q[self.prev_state, self.prev_action])



    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    

   
   



        


        