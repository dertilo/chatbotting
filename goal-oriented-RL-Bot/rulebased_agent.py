import random
from dialogue_config import RULE_REQUESTS, AGENT_ACTIONS, DialogAction


class RuleBasedAgent:
    def __init__(self, eps):

        self.eps = eps
        self.possible_actions = AGENT_ACTIONS
        self.num_actions = len(self.possible_actions)
        self.reset()

    def reset(self):
        self.rule_current_slot_index = 0
        self.rule_phase = "not done"

    def step(self, _):
        if self.eps > random.random():
            action = random.randint(0, self.num_actions - 1)
        else:
            action = self._rule_action()
        return action

    def _rule_action(self) -> int:

        if self.rule_current_slot_index < len(RULE_REQUESTS):
            slot = RULE_REQUESTS[self.rule_current_slot_index]
            self.rule_current_slot_index += 1
            rule_response = DialogAction("request", request_slots={slot: "UNK"})
        elif self.rule_phase == "not done":
            rule_response = DialogAction("match_found")
            self.rule_phase = "done"
        elif self.rule_phase == "done":
            rule_response = DialogAction("done")
        else:
            raise Exception("Should not have reached this clause")

        index = map_action_to_index(rule_response)
        return index


def map_action_to_index(response):
    for (i, action) in enumerate(AGENT_ACTIONS):
        if response == action:
            return i
    raise ValueError("Response: {} not found in possible actions".format(response))


# def map_action_to_index(response):
#     return action2idx[json.dumps(response)]
