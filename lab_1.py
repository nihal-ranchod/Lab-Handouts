###
# Group Members
# Nihal Ranchod - 2427378
# Lisa Godwin - 2437980
# Brendan Griffiths - 2426285
# Konstantinos Hatzipanis - 2444096
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    v = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_s = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v_s += action_prob * prob * (reward + discount_factor * v[next_state])
            delta = max(delta, np.abs(v[s] - v_s))
            v[s] = v_s
        if delta < theta:
            break
    return np.array(v)

def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        expected_action_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, _ in env.P[s][a]:
                expected_action_values[a] += prob * (reward + discount_factor * V[next_state])

        return expected_action_values

    current_policy = (np.ones((*env.shape, env.action_space.n), dtype=np.uint32) / env.action_space.n).reshape(-1,4)

    policy_stable = False 
    while not policy_stable:
        current_value = policy_evaluation_fn(env, current_policy, discount_factor)
        old_policy = np.copy(current_policy)
        for s in range(env.observation_space.n):
            action_value = one_step_lookahead(s, current_value)
            new_action_probs = np.zeros(env.action_space.n)
            new_action_probs[np.argmax(action_value)] = 1.0
            current_policy[s] = new_action_probs
        policy_stable = (old_policy == current_policy).any()
            
    return (current_policy, current_value)




def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        expected_action_values = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, next_state, reward, _ in env.P[s][a]:
                expected_action_values[a] += prob * (reward + discount_factor * V[next_state])

        return expected_action_values

    v = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v_s = np.max(one_step_lookahead(s, v))
            delta = max(delta, np.abs(v[s] - v_s))
            v[s] = v_s
        if delta < theta:
            break

    policy = np.zeros((env.observation_space.n, 4))
    for s in range(env.observation_space.n):
        expected_action_values = one_step_lookahead(s, v)
        policy[s][np.argmax(expected_action_values)] = 1.0
    return policy, v

def generate_random_trajectory(env, policy, state):
    trajectory = []
    done = False

    while not done:
        action = np.random.choice(range(0,4), 1, p=policy[state])[0]
        trajectory.append((state, action))
        _, next_state, _, done = env.P[state][action][0]
        state = next_state

    return trajectory

def generate_trajectory(env, policy, state: int):
    trajectory: list[tuple[int, int]] = []
    done = False

    while not done:
        action: int = np.argmax(policy[state])
        trajectory.append((state, action))
        _, next_state, _, done = env.P[state][action][0]
        state = next_state

    # print("")
    return trajectory

def print_policy(env, policy):
    printed_policy = np.full(env.shape, 'o')
    for s in range(env.observation_space.n):
        row, col = divmod(s, env.shape[1])
        action = np.argmax(policy[s])
        if action == 0:
            printed_policy[row, col] = 'U'  
        elif action == 1:
            printed_policy[row, col] = 'R'  
        elif action == 2:
            printed_policy[row, col] = 'D'  
        elif action == 3:
            printed_policy[row, col] = 'L' 

    printed_policy[env.shape[0] - 1, env.shape[1] - 1] = 'T'
    for row in printed_policy:
        print("  ".join(row))


def print_trajectory(env, trajectory):
    grid_trajectory = np.full(env.shape, 'o')
    for (state, action) in trajectory:
        row, col = divmod(state, env.shape[1])
        if action == 0:
            grid_trajectory[row, col] = 'U'  
        elif action == 1:
            grid_trajectory[row, col] = 'R'  
        elif action == 2:
            grid_trajectory[row, col] = 'D'  
        elif action == 3:
            grid_trajectory[row, col] = 'L' 

    # Mark the terminal state with an 'T'
    grid_trajectory[env.shape[0] - 1, env.shape[1] - 1] = 'T'
    for row in grid_trajectory:
        print("  ".join(row))

def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[24], terminal_reward=0, step_reward=-1)
    initial_state = env.reset()

    # Brendan:
    ## What is a policy: A set of actions for each state
    ## Therefore we can describe our policy as a function (2D array) that maps a state to some actions
    ## Our list of actions is Up (0,-1), Down (0,1), Left (-1,0), Right (1,0)
    ## 
    ## A uniform random policy can be interpreted 1 of 2 ways:
    ## 1. Each state has only one action, which is chosen at random via a uniform distribution
    ##      * Every call of $\pi(s)$ will return the same action $a$
    ##      * $\pi(s)$ has $a$ chosen at initialization time
    ## 2. Each state has 4 action's, that can be chosen at random with a uniform distribution
    ##      * Every call of $\pi(s)$ will return a random action $a$
    ##      * Every call of $\pi(s)$ will choose a new action to return
    ##
    ## By def policy_evaluation:
    ## for a, action_prob in enumerate(policy[s]):
    ##
    ## A policy is defined as a list of probabilites where the index is the action
    ## So [1., 0., 0., 0.] is a policy for a state that has a 100% chance of picking the UP action
    ## [0.25, 0.25, 0.25, 0.25] is a policy for a state that has an equal chance of picking any action

    env.render()
    print("")

    # Exercise 1.1: Generating a trajectory with a uniform random policy
    random_policy = (np.ones((*env.shape, env.action_space.n), dtype=np.uint32) / env.action_space.n).reshape(-1,4)

    trajectory = generate_random_trajectory(env, random_policy, initial_state)
    print_trajectory(env, trajectory)

    print("")
    print("")
    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    random_v = policy_evaluation(env=env, policy=random_policy)
    print("Random Policy Value Function")
    print(random_v)
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.0])
    np.testing.assert_array_almost_equal(random_v, expected_v, decimal=2)

    print("")
    print("")
    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    policy, v = policy_iteration(env)
    print("Policy Iteration Policy")
    print_policy(env, policy)
    print("Policy Iteration Value Function")
    print(v.reshape(env.shape))
    print("Policy Iteration Trajectory")
    print_trajectory(env, generate_trajectory(env, policy, initial_state))
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("")
    print("")
    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")

    policy, v = value_iteration(env)
    print("Value Iteration Policy")
    print_policy(env, policy)
    print("Value Iteration Value Function")
    print(v.reshape(env.shape))
    print("Value Iteration Trajectory")
    print_trajectory(env, generate_trajectory(env, policy, initial_state))
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


    ## Graph Nonsense
    print("")
    print("")
    print("*" * 5 + " Discount Rate Testing " + "*" * 5)
    print("")
    discount_rates = np.logspace(-0.2, 0, num=30)
    pi_times = np.zeros(30)
    vi_times = np.zeros(30)
    for i, gamma in enumerate(discount_rates):
        initial_state = env.reset()

        pi_start = timeit.default_timer()
        pi_policy, pi_value = policy_iteration(env, discount_factor=gamma)
        pi_end = timeit.default_timer()
        pi_times[i] = pi_end - pi_start

        vi_start = timeit.default_timer()
        vi_policy, vi_value = value_iteration(env, discount_factor=gamma)
        vi_end = timeit.default_timer()
        vi_times[i] = vi_end - vi_start

        # print(f"{i}: $\gamma$ = {gamma} | s_i = {initial_state} | Policy Iteration Time = {pi_times[i]} s | Value Iteration Time = {vi_times[i]} s")

    df = pd.DataFrame(data={
        "Discount Factors": discount_rates,
        "Policy Iteration": pi_times * 1000.,
        "Value Iteration": vi_times * 1000.,
    })
    print(df)
    df = df.melt(id_vars="Discount Factors", var_name="Algorithms", value_name="Time (milliseconds)")
    ax = sns.lineplot(data=df, x="Discount Factors", y="Time (milliseconds)", hue="Algorithms", dashes=False, style="Algorithms", markers=True, palette="Set2")
    ax.set_title("PI and VI runtimes over varying discount factors $\gamma$")
    plt.savefig("algorithm_runtimes.pdf")


if __name__ == "__main__":
    main()
