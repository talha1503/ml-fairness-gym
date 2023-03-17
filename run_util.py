# coding=utf-8
# Copyright 2022 The ML Fairness Gym Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Utilities for running and measuring gym simulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from rewards import qlearning_reward_function, update_equalized_group_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import flags
import gin
import tqdm
import numpy as np
import random
from collections import deque

FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_tqdm", True,
                     "Use tqdm to visually represent progress in simulations.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQNAgent(nn.Module):
    # Try out xavier initialization
    def __init__(self, state_space, action_space):
        super().__init__()
        self.layer1 = nn.Linear(state_space, 24)
        self.layer2 = nn.Linear(24, 12)

        self.summary_layer1 = nn.Linear(state_space, 24)
        self.summary_layer2 = nn.Linear(24, 12)

        self.layer3 = nn.Linear(12, action_space)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.tensor(self.x[index], device=device, dtype=torch.float32)
        y = torch.tensor(self.y[index], device=device, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.x)


def train_network(model, x, y, batch_size):
    model = model.to(device)
    dataset = ListDataset(x, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    epochs = 100
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0
        for i, (x, y) in enumerate(dataloader, 0):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        epoch_loss = (epoch_loss / (i + 1))
        # if epoch % 10 == 0:
        #     # epoch_loss = (epoch_loss/(i+1))
        #     print("Epoch: {} Training Loss: {} Training Accuracy: {}".format(epoch, epoch_loss, epoch_accuracy))
    return model


def train_using_bellman_eq(env, replay_memory, model, target_model, done):
    learning_rate = 0.7  # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return model

    batch_size = 64
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = torch.tensor([transition[0] for transition in mini_batch], device=device, dtype=torch.float32)
    current_qs_list = model(current_states)
    new_current_states = torch.tensor([transition[3] for transition in mini_batch], device=device, dtype=torch.float32)
    future_qs_list = target_model(new_current_states)

    x, y = [], []
    for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * torch.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        x.append(state)
        y.append(current_qs)
    model = train_network(model, x, y, batch_size)
    return model


@gin.configurable
def run_simulation(env, agent, metrics, num_steps, rl_agent, seed=100, agent_seed=50):
    """Perform a simple simulation and return a measurement.

  Args:
    env: A `core.FairnessEnv`.
    agent: A `core.Agent`.
    metrics: A list of `core.Metric` instances, a dict of {name: `core.Metric`}
      or a single `core.Metric` instance.
    num_steps: An integer indicating the number of steps to simulate in each
      episode.
    seed: An integer indicating a random seed.
    agent_seed: An integer indicating a random seed for the agent.
    rl_agent: qlearning, dqn, or none to choose from the agents.

  Returns:
    A list of measurements if multiple metrics else a single measurement for a
    single metric.
  """
    # agent.seed(agent_seed)
    env.seed(seed)
    observation = env.reset()
    done = False

    print("Starting simulation")
    simulation_iterator = tqdm.trange if FLAGS.use_tqdm else range
    if rl_agent == 'dqn':
        epsilon = 0.1

        model = DQNAgent(len(env.observation_space['applicant_features'].nvec), 2)
        target_model = DQNAgent(len(env.observation_space['applicant_features'].nvec), 2)

        target_model.load_state_dict(model.state_dict())
        replay_memory = deque(maxlen=50_000)

        steps_to_update_target_model = 0

        model = model.to(device)
        target_model = target_model.to(device)

        for _ in simulation_iterator(num_steps):
            state = env.reset()
            prev_bank_cash = state['bank_cash']
            one_hot_state = state['applicant_features']
            state = np.argmax(state['applicant_features'])

            '''
                Actions:
                Reject: 0
                Accept: 1 
            '''
            equalized_group_dict = {
                'tp_0': 0,
                'tp_1': 0,
                'fn_0': 0,
                'fn_1': 0
            }
            done = False
            for i in range(100):
                steps_to_update_target_model += 1
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    state_reshaped = torch.tensor(one_hot_state, device=device, dtype=torch.float32).unsqueeze(0)
                    prediction = model(state_reshaped).squeeze(0)
                    action = torch.argmax(prediction).item()

                next_state, reward, done, _ = env.step(action)
                current_bank_cash = next_state['bank_cash']
                one_hot_next_state = next_state['applicant_features']
                next_state = np.argmax(next_state['applicant_features'])


                equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                                   env.state.will_default, action)
                reward = qlearning_reward_function(env, action, prev_bank_cash, current_bank_cash, equalized_group_dict)

                replay_memory.append([one_hot_state, action, reward, one_hot_next_state, done])
                ####
                if steps_to_update_target_model % 4 == 0:
                    model = train_using_bellman_eq(env, replay_memory, model, target_model, done)

                one_hot_state = one_hot_next_state
                prev_bank_cash = current_bank_cash

                if done:
                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break

    elif rl_agent == 'qlearning':
        q_table = np.zeros([len(env.observation_space['applicant_features'].nvec), 2])
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        for _ in simulation_iterator(num_steps):
            state = env.reset()
            prev_bank_cash = state['bank_cash']
            state = np.argmax(state['applicant_features'])

            '''
      Actions:
        Reject: 0
        Accept: 1 
      '''

            equalized_group_dict = {
                'tp_0': 0,
                'tp_1': 0,
                'fn_0': 0,
                'fn_1': 0
            }
            for i in range(1000):
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_table[state])

                next_state, reward, done, _ = env.step(action)
                current_bank_cash = next_state['bank_cash']
                next_state = np.argmax(next_state['applicant_features'])

                equalized_group_dict = update_equalized_group_dict(equalized_group_dict, env.state.group_id,
                                                                   env.state.will_default, action)
                reward = qlearning_reward_function(env, action, prev_bank_cash, current_bank_cash, equalized_group_dict)

                old_value = q_table[state, action]
                next_max = np.max(q_table[next_state])
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                q_table[state, action] = new_value
                state = next_state
                prev_bank_cash = current_bank_cash

                if done:
                    break
    else:
        for _ in simulation_iterator(num_steps):
            # Update the agent with any changes to the observation or action space.
            agent.action_space, agent.observation_space = (env.action_space,
                                                           env.observation_space)
            action = agent.act(observation, done)
            # TODO(): Remove reward from this loop.
            observation, reward, done, _ = env.step(action)
            if done:
                break

    print("Measuring metrics")
    if isinstance(metrics, list):
        return [metric.measure(env) for metric in metrics]
    elif isinstance(metrics, dict):
        return {name: metric.measure(env) for name, metric in metrics.items()}
    else:
        return metrics.measure(env)


@gin.configurable
def run_stackelberg_simulation(env,
                               agent,
                               metrics,
                               num_steps,
                               seed=100,
                               agent_seed=100):
    """Performs a Stackelberg simulation.


  A Stackelberg Simulation involves a two player game between a Jury (Agent) and
  Contestants (Environment's population). In this setup the game proceeds as
  follows:
  1. Agent Publishes a classifier
  2. Contestants manipualte features to game the classifier
  3. Agent receives manipulated features and makes decision
  4. Environment receives agent's decision and calculates penalties/reward.

  In this case, we have folded steps 2, 3, 4 into the environment, where once
  the agent publishes its classifier, the feature manipulation, classification
  and reward calculation is done in one step in the environment.

  Args:
    env: A `core.FairnessEnv`.
    agent: A `core.Agent`.
    metrics: A list of `core.Metric` instances, a dict of {name: `core.Metric`}
      or a single `core.Metric` instance.
    num_steps: An integer indicating the numnber of steps to simulate.
    seed: An integer indicating a random seed.
    agent_seed: An integer indicating a random seed for the agent.

  Returns:
    A list of measurements if multiple metrics else a single measurement.
  """
    env.seed(seed)
    agent.seed(agent_seed)
    _ = env.reset()
    agent.action_space = env.action_space
    action = agent.initial_action()
    done = False
    print("Starting simulation")
    simulation_iterator = tqdm.trange if FLAGS.use_tqdm else range
    for _ in simulation_iterator(num_steps):
        # TODO(): Remove reward from this loop.
        observation, _, done, _ = env.step(action)
        # Update the agent with any changes to the observation or action space.
        agent.action_space, agent.observation_space = (env.action_space,
                                                       env.observation_space)
        action = agent.act(observation, done)
        if done:
            break

    print("Measuring metrics")
    if isinstance(metrics, list):
        return [metric.measure(env) for metric in metrics]
    elif isinstance(metrics, dict):
        return {name: metric.measure(env) for name, metric in metrics.items()}
    else:
        return metrics.measure(env)
