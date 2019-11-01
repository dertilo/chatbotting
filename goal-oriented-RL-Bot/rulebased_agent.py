import random
import torch

from dialogue_config import RULE_REQUESTS, AGENT_ACTIONS, map_action_to_index, \
    map_index_to_action


class RuleBasedAgent:

    def __init__(self, eps):

        self.eps = eps
        self.possible_actions = AGENT_ACTIONS
        self.num_actions = len(self.possible_actions)
        self.reset()


    def reset(self):
        self.rule_current_slot_index = 0
        self.rule_phase = "not done"

    def step_single(self, _):
        if self.eps > random.random():
            action =  random.randint(0, self.num_actions - 1)
        else:
            action =  self._rule_action()
        return torch.tensor(action,dtype=torch.int64)

    def _rule_action(self):

        if self.rule_current_slot_index < len(RULE_REQUESTS):
            slot = RULE_REQUESTS[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = {
                "intent": "request",
                "inform_slots": {},
                "request_slots": {slot: "UNK"},
            }
        elif self.rule_phase == "not done":
            rule_response = {
                "intent": "match_found",
                "inform_slots": {},
                "request_slots": {},
            }
            self.rule_phase = "done"
        elif self.rule_phase == "done":
            rule_response = {"intent": "done", "inform_slots": {}, "request_slots": {}}
        else:
            raise Exception("Should not have reached this clause")

        index = map_action_to_index(rule_response)
        return index