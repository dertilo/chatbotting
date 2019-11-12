import random
from typing import Dict, List

from dialogue_config import usersim_intents, DialogAction


class ErrorModelController:
    def __init__(self, slot2values: Dict[str, List[str]], emc_params):

        self.slot2values = slot2values
        self.slot_error_prob = emc_params["slot_error_prob"]
        self.slot_error_mode = emc_params["slot_error_mode"]  # [0, 3]
        self.intent_error_prob = emc_params["intent_error_prob"]
        self.intents = usersim_intents

    def infuse_error(self, action: DialogAction):
        """
        Takes a semantic frame/action as a dict and adds 'error'.

        Given a dict/frame it adds error based on specifications in constants. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.

        Parameters:
            frame (dict): format dict('intent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
        """

        informs_dict = action.inform_slots
        for key in list(action.inform_slots.keys()):
            assert key in self.slot2values
            if random.random() < self.slot_error_prob:
                if self.slot_error_mode == 0:  # replace the slot_value only
                    self._slot_value_noise(key, informs_dict)
                elif self.slot_error_mode == 1:  # replace slot and its values
                    self._slot_noise(key, informs_dict)
                elif self.slot_error_mode == 2:  # delete the slot
                    self._slot_remove(key, informs_dict)
                else:  # Combine all three
                    rand_choice = random.random()
                    if rand_choice <= 0.33:
                        self._slot_value_noise(key, informs_dict)
                    elif rand_choice > 0.33 and rand_choice <= 0.66:
                        self._slot_noise(key, informs_dict)
                    else:
                        self._slot_remove(key, informs_dict)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            action.intent = random.choice(self.intents)

    def _slot_value_noise(self, key, informs_dict):
        """
        Selects a new value for the slot given a key and the dict to change.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict[key] = random.choice(self.slot2values[key])

    def _slot_noise(self, key, informs_dict):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for this new slot.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
        random_slot = random.choice(list(self.slot2values.keys()))
        informs_dict[random_slot] = random.choice(self.slot2values[random_slot])

    def _slot_remove(self, key, informs_dict):
        """
        Removes the slot given the key from the informs dict.

        Parameters:
            key (string)
            informs_dict (dict)
        """

        informs_dict.pop(key)
