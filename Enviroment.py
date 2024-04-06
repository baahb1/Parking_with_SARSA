import numpy as np

class parking_lot:
    class parking_spot:
        def __init__(self,left_spot,Right_spot):
            self.left_spot = left_spot
            self.right_spot = Right_spot
    
    class agent:
        def __init__(self,rows,columns):
            self.row = rows
            self.column = columns


    def __init__(self,rows,columns):
        self.rows = rows
        self.columns = columns
        self.spaces = np.empty(shape=[rows,columns],dtype=self.parking_spot)
        self.agent_O = self.agent(rows,columns)



