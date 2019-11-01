import json
import pickle
from collections import Iterator
from typing import NamedTuple, List, Dict

import gym
import numpy as np
import torch
import torch.nn as nn

from dialogue_config import map_index_to_action
from error_model_controller import ErrorModelController
from state_tracker import StateTracker
from user_simulator import UserSimulator
from utils import remove_empty_slots


class UserGoal(NamedTuple):
    request_slots: dict
    diaact: str
    inform_slots: dict


def mix_in_some_random_actions(policy_actions, eps, num_actions):
    if eps > 0.0:
        random_actions = torch.randint_like(policy_actions, num_actions)
        selector = torch.rand_like(random_actions, dtype=torch.float32)
        actions = torch.where(selector > eps, policy_actions, random_actions)
    else:
        actions = policy_actions
    return actions


class DialogManagerAgent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.num_actions = action_space.n
        n_hid = 32
        self.nn = nn.Sequential(
            *[
                nn.Linear(obs_space.shape[0], n_hid),
                nn.ReLU(),
                # nn.Linear(8, 4),
                # nn.ReLU(),
                nn.Linear(n_hid, self.num_actions),
            ]
        )

    def calc_q_values(self, obs_batch):
        observation_tensor = torch.tensor(obs_batch, dtype=torch.float)
        q_values = self.nn(observation_tensor)
        return q_values

    def step_batch(self, obs_batch, eps=0.001):
        q_values = self.calc_q_values(obs_batch)
        policy_actions = q_values.argmax(dim=1)
        actions = mix_in_some_random_actions(policy_actions, eps, self.num_actions)
        return actions

    def step_single(self, obs, eps=0.001):
        obs_batch = np.expand_dims(obs, 0)
        actions = self.step_batch(obs_batch, eps)
        return int(actions.numpy()[0])


class DialogEnv(gym.Env):
    def __init__(
        self,
        user_goals: List[UserGoal],
        emc_params: Dict,
        max_round_num: int,
        database: Dict,
    ) -> None:

        self.user = UserSimulator(user_goals, max_round_num, database)
        self.emc = ErrorModelController(db_dict, emc_params)
        self.state_tracker = StateTracker(database, max_round_num)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.multi_binary.MultiBinary(
            self.state_tracker.get_state_size()
        )

    def step(self, agent_action):
        self.state_tracker.update_state_agent(agent_action)
        user_action, reward, done, success = self.user.step(agent_action)
        if not done:
            self.emc.infuse_error(user_action)
        self.state_tracker.update_state_user(user_action)
        next_state = self.state_tracker.get_state(done)
        return next_state, reward, done, success

    def reset(self):
        self.state_tracker.reset()
        init_user_action = self.user.reset()
        self.emc.infuse_error(init_user_action)
        self.state_tracker.update_state_user(init_user_action)
        return self.state_tracker.get_state()


def experience_generator(
    agent: DialogManagerAgent, dialog_env: DialogEnv, num_max_steps=30
):
    while True:
        state = dialog_env.reset()
        for turn in range(1, num_max_steps + 1):
            agent_action_index = agent.step_single(state)
            agent_action = map_index_to_action(agent_action_index)
            next_state, reward, done, success = dialog_env.step(agent_action)

            yield {
                "obs": state,
                "next_obs": next_state,
                "action": agent_action_index,
                "next_reward": reward,
                "next_done": done,
            }

            state = next_state
            if done:
                break


def gather_experience(experience_iter: Iterator, batch_size: int = 32):
    experience_batch = [next(experience_iter) for _ in range(batch_size)]
    exp_arrays = {
        key: np.array([exp[key] for exp in experience_batch])
        for key in experience_batch[0].keys()
    }
    return exp_arrays


def get_params(params_json_file="constants.json"):
    with open(params_json_file) as f:
        constants = json.load(f)
    return constants


if __name__ == "__main__":
    params = get_params()
    file_path_dict = params["db_file_paths"]
    DATABASE_FILE_PATH = file_path_dict["database"]
    DICT_FILE_PATH = file_path_dict["dict"]
    USER_GOALS_FILE_PATH = file_path_dict["user_goals"]

    train_params = params["run"]

    database = pickle.load(open(DATABASE_FILE_PATH, "rb"), encoding="latin1")
    remove_empty_slots(database)

    db_dict = pickle.load(open(DICT_FILE_PATH, "rb"), encoding="latin1")
    user_goals = [
        UserGoal(**d)
        for d in pickle.load(open(USER_GOALS_FILE_PATH, "rb"), encoding="latin1")
    ]

    dialog_env = DialogEnv(
        user_goals, params["emc"], params["run"]["max_round_num"], database
    )

    agent = DialogManagerAgent(dialog_env.observation_space, dialog_env.action_space)
    experience_iterator = iter(experience_generator(agent, dialog_env))
    batch = gather_experience(experience_iterator)
    print()
