from gym import spaces
import torch
import numpy as np

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = ("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        # TODO: Initialise agent's networks, optimiser and replay buffer
        self.network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma

        self.target_network.eval()

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO - DONE
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss
    
        self.network.train()

        minibatch = self.replay_buffer.sample(self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = minibatch
        batch_states = torch.Tensor(batch_states).to(device)
        batch_actions = torch.Tensor(batch_actions).to(device=device, dtype=torch.int64)
        batch_rewards = torch.Tensor(batch_rewards).to(device)
        batch_next_states = torch.Tensor(batch_next_states).to(device)
        batch_done = torch.Tensor(batch_done).to(device).float()

        batch_targets = batch_rewards + (1. - batch_done) * self.gamma * torch.max(self.target_network(batch_next_states), dim=1).values
        batch_qvals = torch.gather(self.network(batch_states), 1, batch_actions.unsqueeze(1)).squeeze(1)
       
        loss = self.loss_fn(batch_qvals, batch_targets)
        loss.backward()

        self.optimiser.step()
        self.optimiser.zero_grad()

        return loss

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # TODO update target_network parameters with policy_network parameters - DONE
        self.target_network.load_state_dict(self.network.state_dict())

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state - DONE
        self.network.eval()
        state = np.array(state)
        state = torch.Tensor(state).to(device).reshape((1, 4, 84, 84)) # network expects a batch, but we are passin in only the one state 
        state_quality = self.network(state).detach()
        return torch.argmax(state_quality).cpu()
        
