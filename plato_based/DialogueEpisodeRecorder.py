"""
Copyright (c) 2019 Uber Technologies, Inc.

Licensed under the Uber Non-Commercial License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at the root directory of this project. 

See the License for the specific language governing permissions and
limitations under the License.
"""
from dataclasses import dataclass
from typing import List

from Action import DialogueAct
from State import SlotFillingDialogueState

__author__ = "Alexandros Papangelis"

from copy import deepcopy

import pickle
import os
import datetime

"""
The DialogueEpisodeRecorder is responsible for keeping track of the dialogue 
experience. It has some standard fields and provides a custom field for any 
other information we may want to keep track of.
"""


@dataclass
class TurnState:
    state: SlotFillingDialogueState = None
    action: List[DialogueAct] = None
    reward: float = None
    success: bool = None


@dataclass
class Experience:
    state: SlotFillingDialogueState
    new_state: SlotFillingDialogueState
    action: List[DialogueAct] = None
    reward: float = None
    success: str = None
    cumulative_reward: float = None
    input_utterance: str = None
    output_utterance: str = None
    custom: str = None


class DialogueEpisodeRecorder:
    def __init__(self, size=None, path=None):
        self.dialogues: List[List[Experience]] = []
        self.size = size
        self.current_dialogue: List[Experience] = None
        self.cumulative_reward = 0
        self.path = path

        if path:
            self.load(path)

    def set_path(self, path):
        self.path = path

    def record(
        self,
        new_state,
        turnstate: TurnState,
        input_utterance=None,
        output_utterance=None,
        force_terminate=False,
        custom=None,
    ):
        # TODO: what does len(actions)==0 mean ??
        self.cumulative_reward += turnstate.reward

        # Check if a dialogue is starting or ending
        if self.current_dialogue is None:
            self.current_dialogue = []

        self.current_dialogue.append(
            Experience(
                state=deepcopy(turnstate.state),
                new_state=deepcopy(new_state),
                action=deepcopy(turnstate.action),
                reward=deepcopy(turnstate.reward),
                input_utterance=deepcopy(input_utterance) if input_utterance else "",
                output_utterance=deepcopy(output_utterance) if output_utterance else "",
                success="",
                cumulative_reward=deepcopy(self.cumulative_reward),
                custom=deepcopy(custom) if custom else "",
            )
        )

        if turnstate.state.is_terminal() or force_terminate:
            if turnstate.success is not None:
                self.current_dialogue[-1].success = turnstate.success

            # Check if maximum size has been reached
            if self.size and len(self.dialogues) >= self.size:
                self.dialogues = self.dialogues[(len(self.dialogues) - self.size + 1) :]

            self.dialogues.append(self.current_dialogue)
            self.current_dialogue = []
            self.cumulative_reward = 0

    def save(self, path=None):

        if not path:
            path = self.path

        if not path:
            path = f"Logs/Dialogues{datetime.datetime.now().isoformat()}.pkl"
            print("No Log file name provided. Using default: {0}".format(path))

        obj = {"dialogues": self.dialogues}

        try:
            with open(path, "wb") as file:
                pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

        except IOError:
            raise IOError(
                "Dialogue Episode Recorder I/O Error when " "attempting to save!"
            )

    def load(self, path):

        if not path:
            print(
                "WARNING! Dialogue Episode Recorder: No Log file provided "
                "to load from."
            )

        if self.dialogues:
            print(
                "WARNING! Dialogue Episode Recorder is not empty! Loading "
                "on top of existing experience."
            )

        if isinstance(path, str):
            if os.path.isfile(path):
                print(f"Dialogue Episode Recorder loading dialogues from " f"{path}...")

                with open(path, "rb") as file:
                    obj = pickle.load(file)

                    if "dialogues" in obj:
                        self.dialogues = obj["dialogues"]

                    print("Dialogue Episode Recorder loaded from {0}.".format(path))

            else:
                print(
                    "Warning! Dialogue Episode Recorder Log file %s not " "found" % path
                )
        else:
            print(
                "Warning! Unacceptable value for Dialogue Episode Recorder "
                "Log file name: %s " % path
            )
