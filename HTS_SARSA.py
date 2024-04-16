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
        self.num_states = (rows * columns)
        self.enviroment = enviroment

        self.rand_generator = np.random.RandomState(0)

        self.q = np.zeros(shape=(self.num_states,self.num_actions))


    def agent_init(self, agent_init_info):
        """Setup for the agent called when the experiment first starts.
        
        Args:
        agent_init_info (dict), the parameters used to initialize the agent. The dictionary contains:
        {
            num_states (int): The number of states,
            num_actions (int): The number of actions,
            epsilon (float): The epsilon parameter for exploration,
            step_size (float): The step-size,
            discount (float): The discount factor,
        }
        
        """
        # Store the parameters provided in agent_init_info.
        self.num_actions = agent_init_info["num_actions"]
        self.num_states = agent_init_info["num_states"]
        self.epsilon = agent_init_info["epsilon"]
        self.step_size = agent_init_info["step_size"]
        self.discount = agent_init_info["discount"]
        self.rand_generator = np.random.RandomState(agent_init_info["seed"])
        
        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions)) # The array of a    


    def agent_start(self,observation):
        state = observation
        #print("obs",observation)
        #print(np.shape(self.q))
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
    

   




        


        