import numpy as np
from DQN_Online import Train_DQN_Online
from MA_CIQL import Train_MA_CIQL
from MA_CCQL import Train_MA_CCQL
from MA_CIQR import Train_MA_CIQR
from MA_CCQR import Train_MA_CCQR

def train(Model,Dev_Coord,Risky_region):
    # Generate offline dataset from online DQN agent
    Train_DQN_Online(Dev_Coord,Risky_region)
    
    # Train offline agents
    if(Model=="CIQL"):
        alpha = 1
        Train_MA_CIQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="CCQL"):
        alpha = 1
        Train_MA_CCQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="DIQN"):
        alpha = 0
        Train_MA_CIQL(Model,Dev_Coord,Risky_region,alpha)
    elif(Model=="DCQN"):
        alpha = 0
        Train_MA_CCQL(Model,Dev_Coord,Risky_region,alpha)

    elif(Model=="CIQR"):
        alpha = 1
        eta = 1
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CCQR"):
        alpha = 1
        eta = 1
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="QR-DIQN"):
        alpha = 0
        eta = 1
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="QR-DCQN"):
        alpha = 0
        eta = 1
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CIQR-CVaR"):
        alpha = 1
        eta = 0.15
        Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta)
    elif(Model=="CCQR-CVaR"):
        alpha = 1
        eta = 0.15
        Train_MA_CCQR(Model,Dev_Coord,Risky_region,alpha,eta)
    else:
        print("Wrong Model Name")





Dev_Coord = np.array([[3,1],[7,2],[6,7],[1,6],[7,5],[8,5],[9,1],[6,1],[4,7],[2,3]])

Risky_region = np.array([[3,2],[3,3],[3,4],[3,5],[3,6],
                         [4,2],[4,3],[4,4],[4,5],[4,6],
                         [5,2],[5,3],[5,4],[5,5],[5,6],
                         [6,2],[6,3],[6,4],[6,5],[6,6]])

Model = "CCQR"

if __name__ == "__main__":
    train(Model,Dev_Coord,Risky_region)
