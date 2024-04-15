import Enviroment
import HTS_SARSA
import matplotlib.pyplot as plt

def main():

    # all the commands to populate the enviroment
    parking_lot_O = Enviroment.parking_lot(6,5,2,-.05)
    parking_lot_O.fill_slots_reward()
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
    parking_lot_O.update_agent_state()


    SARSA_O = HTS_SARSA.SARSA(enviroment=parking_lot_O,step_size =.1,epsilon = .1, rows = 6, columns = 5, discount = 1.0)

    action = SARSA_O.agent_start(0)
    parking_lot_O.update_agent_state()
    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    state = 1
    print(action)
    reward = parking_lot_O.agent_O.take_action(action)
    print(reward,"reward")
    action = SARSA_O.agent_step(reward[0],state)
    state+=1

    parking_lot_O.update_agent_state()
    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    

    
    #print reward map
    #plt.imshow(parking_lot_O.reward_map,cmap='binary')
    #plt.colorbar()
    #plt.show()

    #action = SARSA_O.agent_step(1,reward)
    #print(action)



    all_reward_sums = {} # Contains sum of rewards during episode
    all_state_visits = {}

    num_runs = 100 # The number of runs
    num_episodes = 200 # The number of episodes in each run

    #for episode in range(num_episodes):
        

    

    

    

if __name__ == '__main__':
    main()