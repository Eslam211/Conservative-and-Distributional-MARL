import torch
import random
import numpy as np
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_


def collect_random(env, U, dataset, num_samples=100):
    state = env.reset()
    for _ in range(num_samples):
        action = random.sample(range(0, 5*(env.M+1)), U)
        next_state, reward, done = env.step(state,action)
        
        for u in range(U):
            dataset[u].add(state, action[u], reward[u], next_state, done[u])
            
        state = next_state
        if env.DONE:
            state = env.reset()

            
            
def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-Online", help="Run name, default: CQL-Online")
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes, default: 200")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs, default: 150")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--min_eps", type=float, default=0.01, help="Minimal Epsilon, default: 4")
    parser.add_argument("--eps_frames", type=int, default=1e4, help="Number of steps for annealing the epsilon value to the min epsilon, default: 1e5")
    parser.add_argument("--Batch_online", type=int, default=32, help="Batch size for online RL, default: 32")
    parser.add_argument("--Batch_offline", type=int, default=128, help="Batch size for Offline RL, default: 128")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--M", type=int, default=10, help="Number of devices, default: 10")
    parser.add_argument("--Num_Cells", type=int, default=10, help="Number of cells, default: 10")
    parser.add_argument("--U", type=int, default=2, help="Number of agents, default: 2")
    parser.add_argument("--DELTA", type=int, default=500, help="Power weight in the reward function, default: 500")
    parser.add_argument("--penalty", type=int, default=300, help="Penalty due to risk, default: 300")
    parser.add_argument("--data_size_perc", type=int, default=16, help="Percentage of offline dataset, default: 16")
    parser.add_argument("--prob_of_risk", type=int, default=0.1, help="Probability of risk, default: 0.1")
    parser.add_argument("--PATH", type=str, default="/homedir01/eeldeeb19/Offline_RL/", help="Path for saving, default: /homedir01/")
    
    return parser.parse_args(args=[])


def eval_runs(env, agent, eval_runs=100):
    """
    Makes an evaluation run with the current epsilon
    """
    
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        action = [0] * env.U
        while True:
            for u in range(env.U):
                action_agent = agent[u].get_action(state, 0)
                action[u] = action_agent[0]
                
            next_state, reward, done = env.step(state,action)
            
            rewards += env.Total_reward
            state = next_state
            if env.DONE:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

def eval_runs_dist(env, agent, eval_runs=100):
    """
    Makes an evaluation run with the current epsilon
    """
    
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()
        rewards = 0
        action = [0] * env.U
        while True:
            for u in range(env.U):
                action_agent = agent[u].get_action(state, 0)
                action[u] = action_agent
                
            next_state, reward, done = env.step(state,action)
            
            rewards += env.Total_reward
            state = next_state
            if env.DONE:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)


def prep_dataloader(state_dim,dataset,num_UAVs,batch_size=256):
    
    state_start = str(0)
    state_end = str(state_dim - 1)
    action_start = str(state_dim)
    action_end = str(state_dim + num_UAVs - 1) # 1 --> (num_UAVs - 1)
    reward_col = str(state_dim + num_UAVs -1+1) # 2 --> (num_UAVs -1+1)
    next_state_start = str(state_dim + num_UAVs -1+2) # 3 --> (num_UAVs -1+2)
    next_state_end = str(state_dim + num_UAVs -1+2 + state_dim - 1)
    done_col = str(state_dim + num_UAVs -1+2 + state_dim - 1 + 1)    
    
    
    states_df = dataset.loc[:, state_start : state_end]
    actions_df = dataset.loc[:, action_start:action_end]
    rewards_df = dataset.loc[:,reward_col]
    next_states_df = dataset.loc[:, next_state_start : next_state_end]
    done_df = dataset.loc[:,done_col]

    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.long)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    


    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])
    
    
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader

def loss_update_cent(q1_loss, Net_params, Tar_params,optimizer):
    optimizer.zero_grad()
    q1_loss.backward()
    clip_grad_norm_(Net_params, 1.)
    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update_cent(Net_params, Tar_params, tau=1e-3)
    return q1_loss.detach().item()

def soft_update_cent(Net_params, Tar_params, tau):
    for target_param, local_param in zip(Tar_params, Net_params):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

