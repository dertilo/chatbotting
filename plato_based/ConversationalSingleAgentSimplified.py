import DataBase
import Goal
import Ontology
from Action import DialogueAct
from ConversationalAgent import ConversationalAgent

from AgendaBasedUS import AgendaBasedUS
from ErrorModel import ErrorModel
import DialogueManager_simplified as DialogueManager
from slot_filling_reward_function import SlotFillingReward
from DialogueEpisodeRecorder import DialogueEpisodeRecorder

from copy import deepcopy

import os
import random


def build_domain_settings(configuration):

    domain = configuration["DIALOGUE"]["domain"]
    assert os.path.isfile(configuration["DIALOGUE"]["ontology_path"])

    ontology = Ontology.Ontology(configuration["DIALOGUE"]["ontology_path"])
    assert os.path.isfile(configuration["DIALOGUE"]["db_path"])

    cache_sql_results = False
    database = DataBase.SQLDataBase(
        configuration["DIALOGUE"]["db_path"], cache_sql_results
    )
    return domain, ontology, database


class ConversationalSingleAgent(ConversationalAgent):
    """
    Essentially the dialogue system. Will be able to interact with:

    - Simulated Users via:
        - Dialogue Acts
        - Text

    - Human Users via:
        - Text
        - Speech
        - Online crowd?

    - Data
    """

    def __init__(self, configuration):

        super(ConversationalSingleAgent, self).__init__()

        # There is only one agent in this setting
        self.agent_id = 0

        # Dialogue statistics
        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0
        self.total_dialogue_turns = 0

        self.minibatch_length = 50
        self.train_interval = 10
        self.train_epochs = 10

        self.SAVE_LOG = True

        # The dialogue will terminate after MAX_TURNS (this agent will issue
        # a bye() dialogue act.
        self.MAX_TURNS = 15

        self.dialogue_turn = -1
        self.ontology = None
        self.database = None
        self.domain = None
        self.dialogue_manager = None
        self.user_model = None
        self.user_simulator = None

        self.agent_goal = None
        self.goal_generator = None

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.recorder = DialogueEpisodeRecorder()

        # TODO: Handle this properly - get reward function type from config
        self.reward_func = SlotFillingReward()

        self.digest_configuration(configuration)

        self.dialogue_manager = DialogueManager.DialogueManager(
            configuration,
            self.ontology,
            self.database,
            self.agent_id,
            "system",
            configuration["AGENT_0"]["DM"]["policy"],
        )

    def digest_configuration(self, configuration):

        self.domain, self.ontology, self.database = build_domain_settings(configuration)
        self.build_general_settings(configuration)
        self.user_simulator = AgendaBasedUS(
            goal_generator=Goal.GoalGenerator(self.ontology, self.database, None),
            error_model=ErrorModel(
                self.ontology,
                slot_confuse_prob=0.0,
                op_confuse_prob=0.0,
                value_confuse_prob=0.0,
            ),
        )

    def build_general_settings(self, configuration):
        if "GENERAL" in configuration and configuration["GENERAL"]:
            if "experience_logs" in configuration["GENERAL"]:
                dialogues_path = None
                if "path" in configuration["GENERAL"]["experience_logs"]:
                    dialogues_path = configuration["GENERAL"]["experience_logs"]["path"]

                if "load" in configuration["GENERAL"]["experience_logs"] and bool(
                    configuration["GENERAL"]["experience_logs"]["load"]
                ):
                    if dialogues_path and os.path.isfile(dialogues_path):
                        self.recorder.load(dialogues_path)
                    else:
                        raise FileNotFoundError(
                            "Dialogue Log file %s not found (did you "
                            "provide one?)" % dialogues_path
                        )

                if "save" in configuration["GENERAL"]["experience_logs"]:
                    self.recorder.set_path(dialogues_path)
                    self.SAVE_LOG = bool(
                        configuration["GENERAL"]["experience_logs"]["save"]
                    )

    def initialize(self):

        self.dialogue_episode = 0
        self.dialogue_turn = 0
        self.num_successful_dialogues = 0
        self.num_task_success = 0
        self.cumulative_rewards = 0

        self.dialogue_manager.initialize({})

        self.curr_state = None
        self.prev_state = None
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

    def start_dialogue(self, args=None):

        self.dialogue_turn = 0
        self.user_simulator.initialize()

        self.dialogue_manager.restart({})
        sys_response = [DialogueAct("welcomemsg", [])]

        rew, success = self.process_system_action(sys_response)

        self.recorder.record(
            deepcopy(self.dialogue_manager.get_state()),
            self.dialogue_manager.get_state(),
            sys_response,
            rew,
            success,
        )

        self.dialogue_turn += 1

        self.prev_state = None

        # Re-initialize these for good measure
        self.curr_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_success = None
        self.prev_task_success = None

        self.continue_dialogue()

    def process_system_action(self, sys_response):
        self.user_simulator.receive_input(sys_response)
        rew, success = self.reward_func.calculate(
            state=self.dialogue_manager.get_state(), goal=self.user_simulator.goal
        )
        return rew, success

    def continue_dialogue(self):

        usr_input = self.user_simulator.respond()

        self.dialogue_manager.receive_input(usr_input)

        # Keep track of prev_state, for the DialogueEpisodeRecorder
        # Store here because this is the state that the dialogue manager
        # will use to make a decision.
        self.curr_state = deepcopy(self.dialogue_manager.get_state())

        if self.dialogue_turn < self.MAX_TURNS:
            sys_response = self.dialogue_manager.generate_output()

        else:
            sys_response = [DialogueAct("bye", [])]

        rew, success = self.process_system_action(sys_response)

        if self.prev_state:
            self.recorder.record(
                self.prev_state,
                self.curr_state,
                self.prev_action,
                self.prev_reward,
                self.prev_success,
            )

        self.dialogue_turn += 1

        self.prev_state = deepcopy(self.curr_state)
        self.prev_action = deepcopy(sys_response)
        self.prev_reward = rew
        self.prev_success = success

    def end_dialogue(self):
        """
        Perform final dialogue turn. Train and save models if applicable.

        :return: nothing
        """

        # Record final state
        self.recorder.record(
            self.curr_state,
            self.curr_state,
            self.prev_action,
            self.prev_reward,
            self.prev_success,
        )

        if self.dialogue_manager.is_training():
            if (
                self.dialogue_episode % self.train_interval == 0
                and len(self.recorder.dialogues) >= self.minibatch_length
            ):
                for epoch in range(self.train_epochs):

                    # Sample minibatch
                    minibatch = random.sample(
                        self.recorder.dialogues, self.minibatch_length
                    )

                    self.dialogue_manager.train(minibatch)

        self.dialogue_episode += 1
        self.cumulative_rewards += self.recorder.dialogues[-1][-1]["cumulative_reward"]

        if self.dialogue_turn > 0:
            self.total_dialogue_turns += self.dialogue_turn

        if self.dialogue_episode % 10000 == 0:
            self.dialogue_manager.save()

        # Count successful dialogues
        dialogue_success = self.recorder.dialogues[-1][-1]["success"]
        if dialogue_success:
            self.num_successful_dialogues += int(dialogue_success)

    def terminated(self):
        """
        Check if this agent is at a terminal state.

        :return: True or False
        """

        return self.dialogue_manager.at_terminal_state()
