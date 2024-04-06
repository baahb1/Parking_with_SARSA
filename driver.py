import Enviroment


def main():
    print("hello")
    parking_lot_O = Enviroment.parking_lot(5,3)
    print(parking_lot_O.agent_O.row)
    print(parking_lot_O.agent_O.column)

    

if __name__ == '__main__':
    main()