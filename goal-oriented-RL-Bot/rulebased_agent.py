import random, copy
from dialogue_config import RULE_REQUESTS, agent_actions

class RuleBasedAgent:

    def __init__(self, eps):

        self.eps = eps
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)
        self.reset_rulebased_vars()


    def reset_rulebased_vars(self):
        self.rule_current_slot_index = 0
        self.rule_phase = "not done"

    def get_action(self,_):

        if self.eps > random.random():
            index = random.randint(0, self.num_actions - 1)
            action = self._map_index_to_action(index)
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

        index = self._map_action_to_index(rule_response)
        return index, rule_response

    def _map_action_to_index(self, response):

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError("Response: {} not found in possible actions".format(response))


    def _map_index_to_action(self, index):

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError("Index: {} not in range of possible actions".format(index))