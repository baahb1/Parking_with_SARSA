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
    






    #The agent stores its own location with row,column and has an action set described below
    #Consider the sarsa reward formula
    #Q(S[t],A[t]) = Q(S[t],A[t])+alpha(R[t+1] + Epsilon *Q(S[t+1],A[t+1]) - Q(S[t],A[t]))
    #Each of the actions functions returns the R[t+1] part of the function

    class agent:
        def __init__(self,rows,columns,time_penalty):
            self.row = rows
            self.column = columns
            self.time_penalty = time_penalty


        # ACTION SET ------------------------------------------------------------------
        def mov_up(self):
            if(self.row != 1):
                self.row -= 1
                return self.time_penalty
            return -100
        
        def mov_down(self):
            if(self.row != parking_lot.rows):
                self.row += 1
                return self.time_penalty
            return -100
        
        def mov_left(self):
            if(self.row == parking_lot.rows or self.rows == 1):
                if(self.column != 1):
                    self.column -= 1
                    return self.time_penalty
            return -100

        def mov_right(self):
            if(self.row == parking_lot.rows or self.rows == 1):
                if(self.column != parking_lot.columns):
                    self.column += 1
                    return self.time_penalty
            return -100

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
    #contains a second np array for displaing the reward distribution
    def __init__(self,rows,columns,entrance_column,time_penalty):
        self.time_penalty = time_penalty
        self.entrance = [0,entrance_column]
        self.rows = rows
        self.columns = columns
        self.spaces = np.empty(shape=[rows,columns],dtype=self.parking_spot)
        self.reward_map = np.zeros(shape=[rows,columns],dtype=float)
        self.agent_O = self.agent(rows,columns,time_penalty)



    
    
    def fill_slots_reward(self):
        for r in range(self.rows):
            for c in range(self.columns):

                #formulate reward for a slot as a distance from entrance
                reward = 100 - 20 * (abs((r-1) - self.entrance[0]) + abs(c - self.entrance[1]))
                self.spaces[r][c] = parking_lot(0,0,reward,self.time_penalty)
                self.reward_map[r][c] = reward
        
        #removing reward for parking in top row [Pretend top row is a horizontal street only used for driving on]
        for c in range(self.columns):
            self.spaces[0][c] = parking_lot(0,0,-100,self.time_penalty)
            self.spaces[self.rows-1][c] = parking_lot(0,0,-100,self.time_penalty)
            self.reward_map[0][c] = -100
            self.reward_map[self.rows-1] = -100
            

    



    

