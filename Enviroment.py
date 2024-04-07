import numpy as np
import random

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
        
        def set_spots(self,left,right):
            self.left_spot = left
            self.right_spot = right
    






    #The agent stores its own location with row,column and has an action set described below
    #Consider the sarsa reward formula
    #Q(S[t],A[t]) = Q(S[t],A[t])+alpha(R[t+1] + Epsilon *Q(S[t+1],A[t+1]) - Q(S[t],A[t]))
    #Each of the actions functions returns the R[t+1] part of the function

    class agent:
        def __init__(self,parking_lot_O,rows,columns,time_penalty):
            self.row = rows
            self.column = columns
            self.time_penalty = time_penalty
            self.parking_lot_O = parking_lot_O


        def get_state(self):
            left,right = self.parking_lot_O.get_spaces()[self.row-1][self.column-1].get_spots()
            return self.row,self.column,left,right


        # ACTION SET ------------------------------------------------------------------
        def mov_up(self):
            if(self.row != 1):
                self.row -= 1
                return self.time_penalty
            return -100
        
        def mov_down(self):
            if(self.row != self.parking_lot_O.get_shape()[0]):
                self.row += 1
                return self.time_penalty
            return -100
        
        def mov_left(self):
            if(self.row == self.parking_lot_O.get_shape()[0] or self.row == 1):
                if(self.column != 1):
                    self.column -= 1
                    return self.time_penalty
            return -100

        def mov_right(self):
            if(self.row == self.parking_lot_O.get_shape()[0] or self.row == 1):
                if(self.column != self.parking_lot_O.get_shape()[1]):
                    self.column += 1
                    return self.time_penalty
            return -100

        def park_right(self):
            left,right = self.parking_lot_O.get_spaces()[self.row-1][self.column-1].get_spots()
            if(right == 0):
                return self.parking_lot_O.get_spaces()[self.row-1][self.column-1].get_reward()
            else:
                return -100
            
        def park_left(self):
            left,right = self.parking_lot_O.get_spaces()[self.row-1][self.column-1].get_spots()
            if(left == 0):
                return self.parking_lot_O.get_spaces()[self.row-1][self.column-1].get_reward()
            else:
                return -100



    #Init function for Parking_lot. Does not fill values in for the parking_lots
    #contains a second np array for displaing the reward distribution
    def __init__(self,rows,columns,entrance_column,time_penalty):
        self.time_penalty = time_penalty
        self.entrance = [0,entrance_column]
        self.rows = rows
        self.columns = columns
        self.spaces = np.empty(shape=[rows,columns],dtype=self.parking_spot)
        self.reward_map = np.zeros(shape=[rows,columns],dtype=float)
        
        self.agent_O = self.agent(self,rows,columns,time_penalty)

        self.agent_map = np.zeros(shape=[rows,columns],dtype=int)

        self.agent_map[self.agent_O.row-1][self.agent_O.column-1] = 1



    
    


    #GENERATE CAR SPOT DISTRIBUTION
    def fill_slots_reward(self):
        for r in range(self.rows):
            for c in range(self.columns):

                #formulate reward for a slot as a distance from entrance
                reward = 100 - 20 * (abs((r-1) - self.entrance[0]) + abs(c - self.entrance[1]))
                self.spaces[r][c] = self.parking_spot(0,0,reward)
                self.reward_map[r][c] = reward
        
        #removing reward for parking in top row and bottom row [Pretend top/bottom row is a horizontal street only used for driving on]
        for c in range(self.columns):
            self.spaces[0][c] = self.parking_spot(0,0,-100)
            self.spaces[self.rows-1][c] = self.parking_spot(0,0,-100)
            self.reward_map[0][c] = -100
            self.reward_map[self.rows-1] = -100

    def update_agent_state(self):
        self.agent_map = np.zeros(shape=[self.rows,self.columns],dtype=int)
        self.agent_map[self.agent_O.row-1][self.agent_O.column-1] = 1

    def get_spaces(self):
        return self.spaces
            
    def get_shape(self):
        return self.spaces.shape

    def fill_spots_with_car_seeded(self,seed):
        for r in range(1,self.rows-1):
            for c in range(self.columns):
                self.spaces[r][c].set_spots(seed.pop(),seed.pop()) 

    #generates a new string for fill_slots_ based on 
    def generate_parking_random(self,prop_full):
        RL = [0] * ((self.rows-2) * (self.columns) * 2)
        for x in range(len(RL)):
            if(random.random() < prop_full):
                RL[x] = 1
        return RL
            





    

