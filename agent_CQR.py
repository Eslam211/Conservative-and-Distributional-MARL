import torch
import torch.nn as nn
from networks import CQR_DQN
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import pdb


class CQRAgent():
    def __init__(self, seed, state_size, action_size, alpha, eta, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.TAU = 1e-2
        self.GAMMA = 0.99
        self.BATCH_SIZE = 128
        self.Q_updates = 0
        self.n_step = 1
        self.N = 8
        self.alpha = alpha
        self.eta = eta # CQR ---> eta = 1, CQR-CVaR ---> 0 < eta < 1 
        self.quantile_tau = self.eta * torch.FloatTensor([(2 * i + 1) / (2.0 * self.N) for i in range(0,self.N)]).to(device)
        
        self.qnetwork_local = CQR_DQN(state_size, action_size, hidden_size, seed, self.N).to(device)
        self.qnetwork_target = CQR_DQN(state_size, action_size, hidden_size, seed, self.N).to(device)
        
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-5)

    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_action(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        return action
    
    ################################################ Independent ################################################
    def cql_loss_ind(self, q_values, current_action, current_batch_size):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=2, keepdim=True)
        q_a = q_values.gather(2, current_action.unsqueeze(-1).expand(current_batch_size, self.N, 1))
        return (logsumexp - q_a).sum(1).mean()

    def learn_cqr_ind(self, experiences):
        
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = self.qnetwork_target(next_states).detach().cpu() #.max(2)[0].unsqueeze(1) #(batch_size, 1, N)

        current_batch_size = Q_targets_next.size(0)

        action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)
        
        Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(current_batch_size, self.N, 1)).transpose(1,2)

        
        assert Q_targets_next.shape == (current_batch_size,1, self.N)
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))
        Q_expected = self.qnetwork_local(states).gather(2, actions.unsqueeze(-1).expand(current_batch_size, self.N, 1))
        
        Q_expected_c = self.qnetwork_local(states)
        
        td_error = Q_targets - Q_expected
        assert td_error.shape == (current_batch_size, self.N, self.N), "wrong td error shape"
        huber_l = self.calculate_huber_loss_ind(td_error, 1.0)
        quantil_l = abs(self.quantile_tau -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()
        
        cql1_loss = self.cql_loss_ind(Q_expected_c, actions, current_batch_size)

        
        q1_loss = self.alpha*cql1_loss + 0.5 * loss # CQR ---> alpha = 1, QR ---> alpha = 0

        q1_loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update_ind(self.qnetwork_local, self.qnetwork_target)
        return q1_loss.detach().item()

    def calculate_huber_loss_ind(self,td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.N, self.N), "huber loss has wrong shape"
        return loss
    
    
    def soft_update_ind(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)
            
    ################################################ Centralized ################################################
    
    def cql_loss_cent(self, q_values, current_action, current_batch_size):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=2, keepdim=True)

        q_a = q_values.gather(2, current_action.unsqueeze(-1).expand(current_batch_size, self.N, 1))

        return (logsumexp - q_a).sum(1).mean()

    def learn_cqr_cent(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().cpu() #.max(2)[0].unsqueeze(1) #(batch_size, 1, N)

        self.current_batch_size = Q_targets_next.size(0)
        
        action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)

        
        Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(self.current_batch_size, self.N, 1)).transpose(1,2)

        
        assert Q_targets_next.shape == (self.current_batch_size,1, self.N)
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(2, actions.unsqueeze(-1).expand(self.current_batch_size, self.N, 1))
        
        Q_expected_c = self.qnetwork_local(states)
        
        
        cql1_loss = self.cql_loss_cent(Q_expected_c, actions, self.current_batch_size)
        
        return cql1_loss, Q_expected, Q_targets
        
        
    def loss_calc_cent(self,cql1_loss, Q_expected, Q_targets):
        
        td_error = Q_targets - Q_expected
        
        assert td_error.shape == (self.current_batch_size, self.N, self.N), "wrong td error shape"
        huber_l = self.calculate_huber_loss_cent(td_error, 1.0)
        quantil_l = abs(self.quantile_tau -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()

        q1_loss = self.alpha*cql1_loss + 0.5 * loss # alpha = 1
        
        return q1_loss
    
    
    def calculate_huber_loss_cent(self,td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.N, self.N), "huber loss has wrong shape"
        return loss
