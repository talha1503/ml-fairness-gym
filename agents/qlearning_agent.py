import numpy as np
import core


class QLearningAgent(core.Agent):
    def __init__(self, action_space, reward_fn,
                 observation_space):
        super().__init__(action_space, reward_fn,
                         observation_space)

    def initial_action(self):
        pass

    def _act_impl(self, observation, reward, done):
        pass
