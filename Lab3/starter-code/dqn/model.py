from gym import spaces
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"

        self.conv_block = nn.Sequential(
            nn.Conv2d(observation_space.shape[2], 16, 8),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, 4),
            nn.ReLU()
        )
        
        self.mlp_block = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.n)
        )


    def forward(self, x):
        # TODO Implement forward pass
        print(x.shape)
        conv_output = self.conv_block(x)
        print(conv_output.shape)
        mlp_output = self.mlp_block(conv_output)
        return mlp_output
