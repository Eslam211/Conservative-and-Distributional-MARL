
import numpy as np
from collections import deque
import torch
import argparse
from buffer import ReplayBuffer
import glob
from utils import collect_random, get_config, eval_runs, eval_runs_dist, prep_dataloader
import random
from agent_CQR import CQRAgent

import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import math
import copy
import random
import itertools
import torch
from collections import deque
import pandas as pd
from Environment import environment


def Train_MA_CIQR(Model,Dev_Coord,Risky_region,alpha,eta):
    config = get_config()
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = environment(Dev_Coord,Risky_region,config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Episode_Reward = []
    gamma = 0.99


    agent = []

    Num_UAVs_str = str(config.U)
    penalty_str = str(config.penalty)
    data_size_perc_str = str(config.data_size_perc)

    dataset = pd.read_csv(config.PATH+'Datasets/Dataset_Online_DQN_'+data_size_perc_str+'%_'+Num_UAVs_str+'UAVs_pen_'+penalty_str+'.csv')

    dataloader = prep_dataloader(config.U*2 + config.M, dataset, config.U, batch_size=config.Batch_offline)

    for u in range(config.U):
        agent_u = CQRAgent(seed=config.seed, state_size=env.observation_space.shape,
                     action_size=env.action_space.shape[0],
                     alpha=alpha,
                     eta=eta,
                     device=device)
        agent.append(agent_u)


    batches = 0


    eval_reward = eval_runs_dist(env, agent)

    for i in range(1, config.epochs+1):
        loss = [0] * config.U
        for batch_idx, experience in enumerate(dataloader):
            states, actions, rewards, next_states, dones = experience
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            for u in range(config.U):
                loss[u] = agent[u].learn_cqr_ind((states, actions[:,[u]], rewards, next_states, dones))

        if i % config.eval_every == 0:
            eval_reward = eval_runs_dist(env, agent)

            Episode_Reward.append(eval_reward)
            print("Epoch: {} | Reward: {} | Q Loss_: {}".format(i, eval_reward, loss,))
        
    for u in range(config.U):
        u_str = str(u)
        torch.save(agent[u].qnetwork_local.state_dict(), config.PATH+'Saved_Models/'+Model+'_offline_'+data_size_perc_str+'%_UAV_'+u_str+'_pen_'+penalty_str+'.pth')

    
    
