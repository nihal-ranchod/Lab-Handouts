from gym import spaces
import torch.nn as nn
import torch


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

        # self.conv_block = nn.Sequential(
        #     nn.Conv2d(observation_space.shape[0], 16, 7, padding=3),
        #     nn.ReLU(),
        #     nn.AvgPool2d(3,2,padding=1),
            
        #     nn.Conv2d(16, 32, 5, padding=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(3,2,padding=1),
        # )
        
        # self.mlp_block = nn.Sequential(
        #     nn.Linear(14112, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, action_space.n)
        # )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = observation_space.shape[0], out_channels=16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3200, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )

    # take in state, return action values as an array
    def forward(self, state):
        # TODO Implement forward pass - DONE
        # conv_output = self.conv_block(state)
        # if len(conv_output.shape) == 4:
        #     conv_output_linear = conv_output.reshape((conv_output.shape[0],-1,))
        # else:
        #     conv_output_linear = conv_output.reshape((-1,))
        # mlp_output = self.mlp_block(conv_output_linear)
        # return mlp_output

        feature_map = self.block1(state)
        # print(feature_map.shape, torch.flatten(feature_map).shape)

        action_values = self.block2(feature_map)
        # print(action_values.shape)

        return action_values