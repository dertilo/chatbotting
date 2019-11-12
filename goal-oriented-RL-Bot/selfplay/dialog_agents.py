import json
import pickle
import sys
from collections import Iterator
from typing import List, Dict, Any

import gym
import numpy as np
import torch
import torch.nn as nn

from dialogue_config import map_index_to_action, AGENT_ACTIONS
from error_model_controller import ErrorModelController
from rulebased_agent import RuleBasedAgent
from state_tracker import StateTracker
from user_simulator import UserSimulator, UserGoal
from utils import remove_empty_slots


def mix_in_some_random_actions(policy_actions, eps, num_actions):
    if eps > 0.0:
        random_actions = torch.randint_like(policy_actions, num_actions)
        selector = torch.rand_like(random_actions, dtype=torch.float32)
        actions = torch.where(selector > eps, policy_actions, random_actions)
    else:
        actions = policy_actions
    return actions


class DialogAgent(nn.Module):
    def __init__(self, obs_dim, num_actions, n_hid=32):
        # obs_space.shape[0]
        super().__init__()
        self.num_actions = num_actions
        self.exploration_rate = 1.0
        self.nn = nn.Sequential(
            *[nn.Linear(obs_dim, n_hid), nn.ReLU(), nn.Linear(n_hid, self.num_actions)]
        )

    def calc_q_values(self, obs_batch):
        observation_tensor = torch.tensor(obs_batch, dtype=torch.float)
        q_values = self.nn(observation_tensor)
        return q_values

    def step_batch(self, obs_batch):
        q_values = self.calc_q_values(obs_batch)
        policy_actions = q_values.argmax(dim=1)
        actions = mix_in_some_random_actions(
            policy_actions, self.exploration_rate, self.num_actions
        )
        return actions

    def step(self, obs):
        obs_batch = np.expand_dims(obs, 0)
        actions = self.step_batch(obs_batch)
        return int(actions.numpy()[0])

    def reset(self):
        pass
