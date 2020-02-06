import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


def multilayer_perceptron(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Build a feedforward neural network.
    layers = []
    for j in range(len(sizes)-1):
        activation_function = activation if j < len(
            sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), activation_function()]
    return nn.Sequential(*layers)


def train(environment_name='CartPole-v0', hidden_sizes=[32], learning_rate=1e-2,
          epochs=50, batch_size=5000, render=False):

    # make environment, check spaces, get obs / act dims
    environment = gym.make(environment_name)
    assert isinstance(environment.observation_space, Box), \
        "This example only works for environments with continuous state spaces."
    assert isinstance(environment.action_space, Discrete), \
        "This example only works for environments with discrete action spaces."

    observation_dimensions = environment.observation_space.shape[0]
    number_of_actions = environment.action_space.n

    # make core of policy network
    logits_network = multilayer_perceptron(
        sizes=[observation_dimensions]+hidden_sizes+[number_of_actions])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_network(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        log_probability = get_policy(obs).log_prob(act)
        return -(log_probability * weights).mean()

    # make optimizer
    optimizer = Adam(logits_network.parameters(), lr=learning_rate)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_observations = []  # for observations
        batch_actions = []       # for actions
        batch_weights = []       # for R(tau) weighting in policy gradient
        batch_returns = []       # for measuring episode returns
        batch_lengths = []       # for measuring episode lengths

        # reset episode-specific variables
        # first observation comes from starting distribution
        observation = environment.reset()
        done = False            # signal from environment that episode is over
        episode_rewards = []    # list for rewards accrued throughout episode

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                environment.render()

            # save observation
            batch_observations.append(observation.copy())

            # act in the environment
            action = get_action(torch.as_tensor(
                observation, dtype=torch.float32))
            observation, reward, done, _ = environment.step(action)

            # save action, reward
            batch_actions.append(action)
            episode_rewards.append(reward)

            if done:
                # if episode is over, record info about episode
                episode_return, episode_length = sum(
                    episode_rewards), len(episode_rewards)
                batch_returns.append(episode_return)
                batch_lengths.append(episode_length)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [episode_return] * episode_length

                # reset episode-specific variables
                observation, done, episode_rewards = environment.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_observations) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(obs=torch.as_tensor(batch_observations, dtype=torch.float32),
                                  act=torch.as_tensor(
                                      batch_actions, dtype=torch.int32),
                                  weights=torch.as_tensor(
                                      batch_weights, dtype=torch.float32)
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_returns, batch_lengths

    # training loop
    for i in range(epochs):
        batch_loss, batch_returns, batch_lengths = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment_name', '--env',
                        type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(environment_name=args.environment_name,
          render=args.render, learning_rate=args.learning_rate)
