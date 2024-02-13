import torch
import torch.nn as nn
from networks import DDQN
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import pdb


class CQLAgent():
    def __init__(self, state_size, action_size, alpha, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.tau = 1e-3
        self.gamma = 0.99
        self.BATCH_SIZE = 128
        self.alpha = alpha
        

        self.network = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)

        self.target_net = DDQN(state_size=self.state_size,
                            action_size=self.action_size,
                            layer_size=hidden_size
                            ).to(self.device)
        
        
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-4)


    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action

    ################################################ Independent ################################################
    def cql_loss_ind(self, q_values, current_action):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()

    def learn_cql_ind(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = self.network(states)

        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = self.cql_loss_ind(Q_a_s, actions)

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = self.alpha*cql1_loss + 0.5 * bellman_error # alpha = 1
        
        self.optimizer.zero_grad()
        q1_loss.backward()
        clip_grad_norm_(self.network.parameters(), 1.)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update_ind(self.network, self.target_net)
        return q1_loss.detach().item(), cql1_loss.detach().item(), bellman_error.detach().item()

    def soft_update_ind(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    ################################################ Centralized ################################################
    
    def cql_loss_cen(self, q_values, q_a):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
#         q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()

    def learn_cql_cent(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
            
            Q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards/2 + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = self.network(states)

        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = self.cql_loss_cen(Q_a_s, actions)
        
        return cql1_loss, Q_expected, Q_targets
    
    def loss_calc_cent(self,cql1_loss, Q_expected, Q_targets):
                
#         cql1_loss = self.cql_loss(Q_a_s, Q_expected)

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = self.alpha*cql1_loss + 0.5 * bellman_error # alpha = 1
        return q1_loss
    


