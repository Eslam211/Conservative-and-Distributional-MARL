import numpy as np
import random
import itertools
import math


class environment():
    A_max = 100 # max. AOI
    Bo = 10**(-30/10) # channel gain (-30 dB)
    h = 100 # UAV height in meters
    xc = 100 # horizontal distance between cells centers
    yc = 100 # vertical distance between cells centers
    B = 1e6 # bandwidth
    S = 5e6 # packet size
    sigma = (10**(-100/10)) * (10e-3) # noise (-100 dBm)

    steps_mov = 2
    
    v_u = np.matrix([[0, 0], # movement action (N,S,E,W,NE,NW,SE,SW,I)
                   [steps_mov,0],
                   [-steps_mov,0],
                   [0,steps_mov],
                   [0,-steps_mov]])
    length_episode = 100
    
    ###################################################################################################
    def __init__(self, Dev_Coord, Risky_region,config):
        self.Dev_Coord = Dev_Coord
        self.M = config.M
        self.U = config.U
        self.Num_Cells = config.Num_Cells
        self.DELTA = config.DELTA
        self.penalty = config.penalty
        self.prob_of_risk = config.prob_of_risk
        
        self.A_m_max = 100
        self.directions = 5
        
        a1 = list(range(0,self.M+1)) # select one of the devices or select none.
        a3 = list(range(0,self.directions)) # move up, down, right, left or don't move.
        a5 = list([a1,a3])
        self.all_actions = list(itertools.product(*a5))
        self.action_space = np.arange((self.M+1)*self.directions)
        self.observation_space = np.append(np.zeros(self.U*2),np.ones(self.M))
        
        self.Risky_region = Risky_region
    ###################################################################################################
    def reset(self):        
        self.UAV_init_coord = np.array([])
        for i in range(self.U):
            self.init = np.random.randint(self.Num_Cells, size=2)
            while(any(np.array_equal(x, self.init) for x in self.Risky_region)):
                self.init = np.random.randint(self.Num_Cells, size=2) # make sure UAV is out or risky region
            self.UAV_init_coord = np.append(self.UAV_init_coord,self.init)
        
        self.A_m = []
        self.cntt = 1
        self.done = np.zeros(self.U)
        self.DONE = 0
        self.risk_indicator = [0] * self.U
        
        self.A_m = np.ones(self.M) # append number of initial age for all devices
        
            
        return np.concatenate((np.asarray(self.UAV_init_coord).reshape(-1), self.A_m), axis=None)
    
    ###################################################################################################
    def AoI_Calc(self,DEV_chosen,A_m):
        if(DEV_chosen < self.M):
            A_m[DEV_chosen] = 1
        return A_m

    ###################################################################################################
    def Update_Trajec(self,l_U,V_n_Rnd):
        check_0 = 0 # check for exceeding grid coordinates
        l_U = l_U+V_n_Rnd
        for i in range(2):
            l_U[0,i] = max(0,l_U[0,i])
            l_U[0,i] = min(self.Num_Cells-1,l_U[0,i])
        
        l_U = np.asarray(l_U).reshape(-1)
        return l_U
        
    ###################################################################################################
    def Min_Power(self,dev,L_U):
        if(dev<self.M):
            MIN_Rd = self.xc*math.dist(L_U, self.Dev_Coord[dev])
            MIN_PWR = ((MIN_Rd**2+self.h**2)*(2**(self.S/self.B)-1)*self.sigma)/self.Bo
        else:
            MIN_PWR = 0
        return MIN_PWR*self.DELTA
    
    ###################################################################################################
    def Reward_Calc(self,A_m,Pwr):
        reward = 0
        reward = reward-Pwr-(np.sum(A_m)/self.M)
        return reward
    
    ###################################################################################################  
    def Risk_prob(self,l_U):
        if(any(np.array_equal(x, l_U) for x in self.Risky_region)):
            risk = self.prob_of_risk
        else:
            risk = 0
        return risk
    
    ###################################################################################################  
    def step(self,state,Action_all):
        
        self.L_U = state[0:self.U*2]
        self.A_m = state[self.U*2:self.U*2+self.M]
        
        self.A_m = np.minimum(self.A_m+1,self.A_m_max) # age increment
        
        self.Pwr = 0
        deduction = 0
        L_U_new = np.zeros(self.U*2)
        reward = [0]*self.U
        
        
        for u_cntt in range(self.U): # loop for the number of agents
            L_U_ind = self.L_U[u_cntt*2:u_cntt*2+2] # Location of agent i
            action_chosen = Action_all[u_cntt] # Agent i action
            action = self.all_actions[action_chosen]
            action = np.array(action)

            self.dev_chosen = action[0] # served device by agent i
            self.MOV_DIR = action[1] 
            self.v_n_rnd = self.v_u[self.MOV_DIR] # movement direction by agent i
            
            if(self.done[u_cntt]==0):
                L_U_ind = self.Update_Trajec(L_U_ind,self.v_n_rnd) # update UAV trajectory
                
                risk = self.Risk_prob(L_U_ind) # risk probability
                if(risk == 0):
                    Pwr_agent = self.Min_Power(self.dev_chosen,L_U_ind) # pwr calculations for agent i
                    self.Pwr = self.Pwr + Pwr_agent # update total power
                    self.A_m = self.AoI_Calc(self.dev_chosen,self.A_m) # age calculations
                else:
                    self.risk_indicator[u_cntt] = self.risk_indicator[u_cntt] + 1
                    prob = random.random()
                    if(prob < risk):
                        Pwr_agent = self.Min_Power(self.dev_chosen,L_U_ind) # pwr calculations for agent i
                        self.Pwr = self.Pwr + Pwr_agent + self.penalty
                        self.A_m = self.AoI_Calc(self.dev_chosen,self.A_m) # age calculations
                    else:
                        Pwr_agent = self.Min_Power(self.dev_chosen,L_U_ind) # pwr calculations for agent i
                        self.Pwr = self.Pwr + Pwr_agent # update total power
                        self.A_m = self.AoI_Calc(self.dev_chosen,self.A_m) # age calculations
                    
                    
            
            L_U_new[u_cntt*2:u_cntt*2+2] = L_U_ind # update the trajectory
        
        self.Total_reward = self.Reward_Calc(self.A_m,self.Pwr/self.U)
        
        
        for u_cntt in range(self.U):
            reward[u_cntt] = self.Reward_Calc(self.A_m,self.Pwr)

        
        
        state_new = np.concatenate((np.asarray(L_U_new).reshape(-1), self.A_m), axis=None)
        
        if(self.cntt == self.length_episode): # episode length check
            self.done = np.ones(self.U)
        self.cntt = self.cntt + 1
        
        if(sum(self.done) == self.U):
            self.DONE = 1
        else:
            self.DONE = 0
        
        
        return state_new, reward, self.done
    