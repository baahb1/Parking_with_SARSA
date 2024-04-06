import Enviroment
import matplotlib.pyplot as plt

def main():
    print("hello")
    parking_lot_O = Enviroment.parking_lot(5,5,2,-.05)
    print(parking_lot_O.agent_O.row)
    print(parking_lot_O.agent_O.column)
    parking_lot_O.fill_slots_reward()
    

    plt.imshow(parking_lot_O.reward_map,cmap='binary')
    plt.colorbar()
    plt.show()

    

if __name__ == '__main__':
    main()