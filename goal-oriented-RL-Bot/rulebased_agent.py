import random
from dialogue_config import RULE_REQUESTS, AGENT_ACTIONS, map_action_to_index, \
    map_index_to_action


class RuleBasedAgent:

    def __init__(self, eps):

        self.eps = eps
        self.possible_actions = AGENT_ACTIONS
        self.num_actions = len(self.possible_actions)
        self.reset_rulebased_vars()


    def reset_rulebased_vars(self):
        self.rule_current_slot_index = 0
        self.rule_phase = "not done"

    def get_action(self,_):

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = map_index_to_action(index)
            return index, action
        else:
            return self._rule_action()

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
        return index, rule_response