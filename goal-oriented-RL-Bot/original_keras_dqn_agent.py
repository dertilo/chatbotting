from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random, copy
import numpy as np

from Experience import Experience
from dialogue_config import AGENT_ACTIONS, map_index_to_action
import re


# Some of the code based off of https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
# Note: In original paper's code the epsilon is not annealed and annealing is not implemented in this code either


class DQNAgent:

    def __init__(self, state_size, constants):

        self.C = constants["agent"]
        self.eps = self.C["epsilon_init"]
        self.use_ddqn = not self.C["vanilla"]
        self.lr = self.C["learning_rate"]
        self.gamma = self.C["gamma"]
        self.batch_size = self.C["batch_size"]
        self.hidden_size = self.C["dqn_hidden_size"]

        self.load_weights_file_path = self.C["load_weights_file_path"]
        self.save_weights_file_path = self.C["save_weights_file_path"]

        self.state_size = state_size
        self.possible_actions = AGENT_ACTIONS
        self.num_actions = len(self.possible_actions)

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()

        self._load_weights()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation="relu"))
        model.add(Dense(self.num_actions, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    def step(self, state):

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
        else:
            index = np.argmax(self._dqn_predict_one(state))

        return index

    def _dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self._dqn_predict(
            state.reshape(1, self.state_size), target=target
        ).flatten()

    def _dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target:
            return self.tar_model.predict(states)
        else:
            return self.beh_model.predict(states)

    def train(self,experience:Experience):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        # Calc. num of batches to run
        num_batches = len(experience.memory) // self.batch_size
        for b in range(num_batches):
            batch = random.sample(experience.memory, self.batch_size)

            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])

            assert states.shape == (
                self.batch_size,
                self.state_size,
            ), "States Shape: {}".format(states.shape)
            assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict(states)  # For leveling error
            if self.use_ddqn:
                beh_next_states_preds = self._dqn_predict(
                    next_states
                )  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict(
                next_states, target=True
            )  # For target value for DQN (& DDQN)

            inputs = np.zeros((self.batch_size, self.state_size))
            targets = np.zeros((self.batch_size, self.num_actions))

            for i, (s, a, r, s_, d) in enumerate(batch):
                t = beh_state_preds[i]
                if self.use_ddqn:
                    t[a] = r + self.gamma * tar_next_state_preds[i][
                        np.argmax(beh_next_states_preds[i])
                    ] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)

                inputs[i] = s
                targets[i] = t

            self.beh_model.fit(inputs, targets, epochs=1, verbose=0)

    def update_target_model_weights(self):
        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r"\.h5", r"_beh.h5", self.save_weights_file_path)
        self.beh_model.save_weights(beh_save_file_path)
        tar_save_file_path = re.sub(r"\.h5", r"_tar.h5", self.save_weights_file_path)
        self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r"\.h5", r"_beh.h5", self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r"\.h5", r"_tar.h5", self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
