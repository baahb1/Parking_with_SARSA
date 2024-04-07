import Enviroment
import SARSA
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


    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    print(parking_lot_O.agent_O.mov_left())
    parking_lot_O.update_agent_state()


    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    print(parking_lot_O.agent_O.mov_up())
    parking_lot_O.update_agent_state()

    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    print(parking_lot_O.agent_O.park_left())
    parking_lot_O.update_agent_state()

    plt.imshow(parking_lot_O.agent_map,cmap='binary')
    plt.colorbar()
    plt.show()

    

    

    

if __name__ == '__main__':
    main()