import Enviroment
import HTS_SARSA
import matplotlib.pyplot as plt
import numpy as np
import rl_glue


def main():

    # all the commands to populate the enviroment
    parking_lot_O = Enviroment.parking_lot(6,5,2,-.05)
    parking_lot_O.fill_slots_reward()

    #parking slot seeded 40 slot 60% taken
    seed = [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1]
    parking_lot_O.fill_spots_with_car_seeded(seed)


    #print reward map
    #plt.imshow(parking_lot_O.reward_map,cmap='binary')
    #plt.colorbar()
    #plt.show()


    #plt.imshow(parking_lot_O.agent_map,cmap='binary')
    #plt.colorbar()
    #plt.show()

    #print(parking_lot_O.agent_O.mov_left())
    #parking_lot_O.update_agent_state()


    SARSA_O = HTS_SARSA.SARSA(enviroment=parking_lot_O,step_size =.1,epsilon = .1, rows = 6, columns = 5, discount = 1.0)

    action = SARSA_O.agent_start(0)
    print(action)
    reward = parking_lot_O.agent_O.take_action(action)
    

   



    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {}

    num_runs = 100 # The number of runs
    num_episodes = 200 # The number of episodes in each run

    start_SARSA(parking_lot_O,parking_lot_O.agent_O,environment_parameters,agent_parameters,experiment_parameters)

    

    

def start_SARSA(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
        
        all_reward_sums = [] # Contains sum of rewards during episode
        all_state_visits = [] # Contains state visit counts during the last 10 episodes
        num_runs = 100 # The number of runs
        num_episodes = 200 # The number of episodes in each run

        glue = rl_glue(environment,agent)

        # save rmsve at the end of each episode
        agent_rmsve = np.zeros((experiment_parameters["num_runs"],int(experiment_parameters["num_episodes"]/experiment_parameters["episode_eval_frequency"]) + 1))

        # save learned state value at the end of each run
        agent_state_val = np.zeros((experiment_parameters["num_runs"], environment_parameters["num_states"]))

        env_info = {"num_states": (environment.rows * environment.columns * 4) ,
                    "start_state": environment_parameters["start_state"],
                    "left_terminal_state": environment_parameters["left_terminal_state"],
                    "right_terminal_state": environment_parameters["right_terminal_state"]}

        agent_info = {"num_states": (environment.rows * environment.columns * 4),
                        "num_hidden_layer": agent_parameters["num_hidden_layer"],
                        "num_hidden_units": agent_parameters["num_hidden_units"],
                        "step_size": agent_parameters["step_size"],
                        "discount_factor": environment_parameters["discount_factor"],
                        "beta_m": agent_parameters["beta_m"],
                        "beta_v": agent_parameters["beta_v"],
                        "epsilon": agent_parameters["epsilon"]
                        }
        
    
experiment_parameters = {
    "num_runs" : 20,
    "num_episodes" : 1000,
    "episode_eval_frequency" : 10 # evaluate every 10 episode
}

# Environment parameters
environment_parameters = {
    "num_states" : 500,
    "start_state" : 250,
    "left_terminal_state" : 0,
    "right_terminal_state" : 501,
    "discount_factor" : 1.0
}

# Agent parameters
agent_parameters = {
    "num_hidden_layer": 1,
    "num_hidden_units": 100,
    "step_size": 0.001,
    "beta_m": 0.9,
    "beta_v": 0.999,
    "epsilon": 0.0001,
}

if __name__ == '__main__':
    main()