import numpy as np


class parking_lot:
    #Parking lot contains an np array of shape (parking_lot.rows,parking_lot.columns) with elements of type parking_spot
    #Parking lot also contains one Agent and the cordinates for a entrance
    
    #Using matplotlib can visualise your parking lot value distribution with the following code
    #plt.imshow(parking_lot_O.reward_map,cmap='binary')
    #plt.colorbar()
    #plt.show()

    class parking_spot:
        def __init__(self):
            self.left_spot = -1
            self.right_spot = -1
            self.reward = -1

        def __init__(self,left_spot, Right_spot, reward):
            self.left_spot = left_spot
            self.right_spot = Right_spot
            self.reward = reward
        
        def get_spots(self):
            return self.left_spot,self.right_spot
        
        def get_reward(self):
            return self.reward
    





    class agent:
        def __init__(self,rows,columns):
            self.row = rows
            self.column = columns


        # ACTION SET ------------------------------------------------------------------
        def mov_up(self):
            if(self.row != 1):
                self.row -= 1
                return 0
            return 1
        
        def mov_down(self):
            if(self.row != parking_lot.rows):
                self.row += 1
                return 0
            return 1
        
        def mov_left(self):
            if(self.row == parking_lot.rows or self.rows == 1):
                if(self.column != 1):
                    self.column -= 1
                    return 0
            return 1

        def mov_right(self):
            if(self.row == parking_lot.rows or self.rows == 1):
                if(self.column != parking_lot.columns):
                    self.column += 1
                    return 0
            return 1

        def park_right(self):
            left,right = parking_lot.spaces[self.row][self.column].get_spots()
            if(right == 0):
                return parking_lot.spaces[self.row][self.column].get_reward()
            else:
                return -100
            
        def park_left(self):
            left,right = parking_lot.spaces[self.row][self.column].get_spots()
            if(left == 0):
                return parking_lot.spaces[self.row][self.column].get_reward()
            else:
                return -100



    #Init function for Parking_lot. Does not fill values in for the parking_lots
    def __init__(self,rows,columns,entrance_column):
        self.entrance = [0,entrance_column]
        self.rows = rows
        self.columns = columns
        self.spaces = np.empty(shape=[rows,columns],dtype=self.parking_spot)
        self.reward_map = np.zeros(shape=[rows,columns],dtype=float)
        self.agent_O = self.agent(rows,columns)



    
    
    def fill_slots_reward(self):
        for r in range(self.rows):
            for c in range(self.columns):

                #formulate reward for a slot as a distance from entrance
                reward = 100 - (abs(r - self.entrance[0]) + abs(c - self.entrance[1]))
                self.spaces[r][c] = parking_lot(0,0,reward)
                self.reward_map[r][c] = reward

    

