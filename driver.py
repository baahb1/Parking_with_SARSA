import Enviroment
import HTS_SARSA
import matplotlib.pyplot as plt
import numpy as np
import rl_glue
from tqdm import tqdm

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

    #action = SARSA_O.agent_start(0)
    #print(action)
    #reward = parking_lot_O.agent_O.take_action(action)
    

   



    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {}

    num_runs = 100 # The number of runs
    num_episodes = 200 # The number of episodes in each run

    start_SARSA(parking_lot_O,SARSA_O)

    

    

def start_SARSA(environment, agent):
    np.random.seed(0)

    agents = {
    "Expected Sarsa": agent
    }
    env = environment
    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {} # Contains state visit counts during the last 10 episodes
    agent_info = {"num_actions": 6, "num_states": (environment.rows * environment.columns), "epsilon": 0.1, "step_size": 0.5, "discount": 1.0}
    env_info = {}
    num_runs = 400 # The number of runs
    num_episodes = 600 # The number of episodes in each run


    all_reward_sums = []
    all_state_visits = [] 

    for run in tqdm(range(num_runs)):
        agent_info["seed"] = run
        glue = rl_glue.RLGlue(env, agent)
        glue.rl_init(agent_info, env_info)

        reward_sums = []
        state_visits = np.zeros(48)
        for episode in range(num_episodes):
            if episode < num_episodes - 10:
                # Runs an episode
                glue.rl_episode(10000) 
            else: 
                # Runs an episode while keeping track of visited states
                state, action = glue.rl_start()
                state_visits[state] += 1
                is_terminal = False
                while not is_terminal:
                    reward, state, action, is_terminal = glue.rl_step()
                    state_visits[state] += 1
                
            reward_sums.append(glue.rl_return())
            
        all_reward_sums.append(reward_sums)
        all_state_visits.append(state_visits)


        # plot results
    print(np.mean(all_reward_sums[300]))
    plt.plot(np.mean( all_reward_sums , axis=0), label="Sarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of\n rewards\n during\n episode",rotation=0, labelpad=40)
    plt.ylim(-100,100)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()