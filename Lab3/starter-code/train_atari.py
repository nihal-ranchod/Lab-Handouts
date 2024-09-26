import random
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.wrappers import *

if __name__ == "__main__":

    hyper_params = {
        "seed": 42,  # which seed to use
        "env": "PongNoFrameskip-v4",  # name of the game
        "replay-buffer-size": int(5e3),  # replay buffer size
        "learning-rate": 1e-4,  # learning rate for Adam optimizer
        "discount-factor": 0.99,  # discount factor
        "num-steps": int(1e6),  # total number of steps to run the environment for
        # "num-steps": 100000,
        "batch-size": 256,  # number of transitions to optimize at the same time
        "learning-starts": 256,  # number of steps before learning starts
        "learning-freq": 5,  # number of iterations between every optimization step
        "use-double-dqn": True,  # use double deep Q-learning
        "target-update-freq": 1000,  # number of iterations between every target network update
        "eps-start": 1.0,  # e-greedy start threshold
        "eps-end": 0.01,  # e-greedy end threshold
        "eps-fraction": 0.1,  # fraction of num-steps
        "print-freq": 10,
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    env = gym.make(hyper_params["env"])
    env.seed(hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    # TODO Pick Gym wrappers to use - DONE
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)

    replay_buffer = ReplayBuffer(hyper_params["replay-buffer-size"])

    # TODO Create dqn agent
    agent = DQNAgent(env.observation_space, 
                     env.action_space, 
                     replay_buffer, 
                     hyper_params["use-double-dqn"], 
                     hyper_params["learning-rate"],
                     hyper_params["batch-size"],
                     hyper_params["discount-factor"])
    # agent.act(env.observation_space.sample())

    eps_timesteps = hyper_params["eps-fraction"] * float(hyper_params["num-steps"])
    episode_rewards = [0.0]
    episode_loss = [0.0]

    state = env.reset()
    for t in range(hyper_params["num-steps"]):
        # _ = env.step(env.action_space.sample())
        fraction = min(1.0, float(t) / eps_timesteps)
        eps_threshold = hyper_params["eps-start"] + fraction * (
            hyper_params["eps-end"] - hyper_params["eps-start"]
        )
        sample = random.random()
        # TODO - DONE
        #  select random action if sample is less equal than eps_threshold
        # take step in env
        # add state, action, reward, next_state, float(done) to reply memory - cast done to float
        # add reward to episode_reward

        # POSSIBLE SOLUTION
        if random.random() < eps_threshold:
            action = env.action_space.sample()
        else:
            action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state

        episode_rewards[-1] += reward
        if done:
            state = env.reset()
            episode_rewards.append(0.0)
            episode_loss.append(0.0)

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["learning-freq"] == 0
        ):
            loss = agent.optimise_td_loss()
            episode_loss[-1] += loss.cpu().item()

        if (
            t > hyper_params["learning-starts"]
            and t % hyper_params["target-update-freq"] == 0
        ):
            agent.update_target_network()

        num_episodes = len(episode_rewards)

        if (
            done
            and hyper_params["print-freq"] is not None
            and len(episode_rewards) % hyper_params["print-freq"] == 0
        ):
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            print("")
            print("********************************************************")
            print("steps: {}".format(t))
            print("episodes: {}".format(num_episodes))
            print("mean 100 episode reward: {}".format(mean_100ep_reward))
            print("% time spent exploring: {}".format(int(100 * eps_threshold)))
            print("********************************************************")
            
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    axes[0].set_title("Reward per Episode")
    axes[0].plot(np.arange(1, num_episodes + 1), episode_rewards)

    axes[1].set_title("Loss per Episode")
    axes[1].plot(np.arange(1, num_episodes + 1), episode_loss)

    plt.savefig("./graphs.pdf")

    test_env = gym.make(hyper_params["env"], render_mode="rgb_array")
    test_env = NoopResetEnv(test_env, noop_max=30)
    test_env = MaxAndSkipEnv(test_env, skip=4)
    test_env = EpisodicLifeEnv(test_env)
    test_env = FireResetEnv(test_env)
    # TODO Pick Gym wrappers to use - DONE
    test_env = WarpFrame(test_env)
    test_env = PyTorchFrame(test_env)
    test_env = FrameStack(test_env, 4)
    test_env = gym.wrappers.RecordVideo(test_env, video_folder="./", name_prefix="test_video")

    state = test_env.reset()
    test_env.start_video_recorder()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done, _ = test_env.step(action)

    torch.save(agent.network, "./saved_dqn_agent.pth")
    test_env.close_video_recorder()
    test_env.close()

    env.close()
