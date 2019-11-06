from dataclasses import dataclass
from typing import Dict, List, NamedTuple

from dialogue_config import (
    usersim_default_key,
    FAIL,
    NO_OUTCOME,
    SUCCESS,
    usersim_required_init_inform_keys,
    no_query_keys,
    DialogAction,
    USER,
    PLACEHOLDER,
    UNK,
)
from utils import reward_function
import random, copy


class UserGoal(NamedTuple):
    request_slots: dict
    diaact: str
    inform_slots: dict


@dataclass
class DialogState:  # TODO(tilo): is this really needed?
    intent: str
    history_slots: Dict[str, str]
    inform_slots: Dict[str, str]
    request_slots: Dict[str, str]
    rest_slots: Dict[str, str]


class UserSimulator:
    def __init__(self, goal_list: List[UserGoal], max_round: int):

        self.goal_list = goal_list
        self.max_round = max_round
        self.default_key = usersim_default_key
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = usersim_required_init_inform_keys
        self.no_query = no_query_keys

    def reset(self):
        self.goal = random.choice(self.goal_list)
        self.goal.request_slots[self.default_key] = "UNK"
        rest_slots = {**self.goal.inform_slots, **self.goal.request_slots}
        self.state = DialogState("", {}, {}, {}, rest_slots)
        self.constraint_check = FAIL
        return self._return_init_action()

    def _return_init_action(self):
        self.state.intent = "request"

        if self.goal.inform_slots:
            # Pick all the required init. informs, and add if they exist in goal inform slots
            for inform_key in self.init_informs:
                if inform_key in self.goal.inform_slots:
                    self.state.inform_slots[inform_key] = self.goal.inform_slots[
                        inform_key
                    ]
                    self.state.rest_slots.pop(inform_key)
                    self.state.history_slots[inform_key] = self.goal.inform_slots[
                        inform_key
                    ]
            # If nothing was added then pick a random one to add
            if not self.state.inform_slots:
                key, value = random.choice(list(self.goal.inform_slots.items()))
                self.state.inform_slots[key] = value
                self.state.rest_slots.pop(key)
                self.state.history_slots[key] = value

        req_key = self.get_request_key()
        self.state.request_slots[req_key] = "UNK"

        user_response = DialogAction(
            self.state.intent,
            self.state.inform_slots,
            self.state.request_slots,
            speaker=USER,
        )

        return user_response

    def get_request_key(self):
        non_default_slots = [
            k for k in self.goal.request_slots.keys() if k != self.default_key
        ]
        if len(non_default_slots) > 0:
            req_key = random.choice(non_default_slots)
        else:
            req_key = self.default_key
        return req_key

    def step(self, agent_action: DialogAction):

        self.validate_action(agent_action)

        self.state.inform_slots.clear()
        self.state.intent = ""

        done = False
        success = NO_OUTCOME
        # First check round num, if equal to max then fail
        if agent_action.turn == self.max_round:
            done = True
            success = FAIL
            self.state.intent = "done"
            self.state.request_slots.clear()
        else:
            agent_intent = agent_action.intent
            if agent_intent == "request":
                self._response_to_request(agent_action)
            elif agent_intent == "inform":
                self._response_to_inform(agent_action)
            elif agent_intent == "match_found":
                self._response_to_match_found(agent_action.inform_slots)
            elif agent_intent == "done":
                success = self._response_to_done()
                self.state.intent = "done"
                self.state.request_slots.clear()
                done = True

        self.validate_state(self.state)

        user_response = DialogAction(
            self.state.intent,
            self.state.inform_slots,
            self.state.request_slots,
            speaker=USER,
        )

        reward = reward_function(success, self.max_round)

        return user_response, reward, done, True if success is 1 else False

    def validate_state(self, s: DialogState):
        # If request intent, then make sure request slots
        if s.intent == "request":
            assert s.request_slots
        # If inform intent, then make sure inform slots and NO request slots
        if s.intent == "inform":
            assert s.inform_slots
            assert not s.request_slots
        assert "UNK" not in s.inform_slots.values()
        assert "PLACEHOLDER" not in s.request_slots.values()
        # No overlap between rest and hist
        for key in s.rest_slots:
            assert key not in s.history_slots
        for key in s.history_slots:
            assert key not in s.rest_slots
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal.inform_slots:
            assert s.history_slots.get(inf_key, False) or s.rest_slots.get(
                inf_key, False
            )
        for req_key in self.goal.request_slots:
            assert s.history_slots.get(req_key, False) or s.rest_slots.get(
                req_key, False
            ), req_key
        # Anything in the rest should be in the goal
        for key in s.rest_slots:
            assert self.goal.inform_slots.get(
                key, False
            ) or self.goal.request_slots.get(key, False)
        assert s.intent != ""
        # -----------------------

    def validate_action(self, agent_action):
        if agent_action.inform_slots is not None:
            assert all(
                value != UNK and value != PLACEHOLDER
                for value in agent_action.inform_slots.values()
            )
        if agent_action.request_slots is not None:
            assert all(
                value != PLACEHOLDER for value in agent_action.request_slots.values()
            )

    def _response_to_request(self, agent_action: DialogAction):

        agent_request_key = list(agent_action.request_slots.keys())[0]

        if self.agent_requests_slot_that_user_wants_to_inform_about(agent_request_key):
            self.handle_meaningful_agent_request(agent_request_key)
        elif self.agent_requests_what_he_already_informed_about(agent_request_key):
            self.inform_agent_about_what_he_should_know(agent_request_key)
        elif self.agent_asks_for_what_user_wants_to_ask(agent_request_key):
            self.handle_third_case(agent_request_key)
        else:  # agent ask for slot that user does not care about
            self.prepare_dontcare(agent_request_key)

    def agent_requests_slot_that_user_wants_to_inform_about(self, agent_request_key):
        return agent_request_key in self.goal.inform_slots

    def agent_requests_what_he_already_informed_about(self, agent_request_key):
        return (
            agent_request_key in self.goal.request_slots
            and agent_request_key in self.state.history_slots
        )

    def agent_asks_for_what_user_wants_to_ask(self, agent_request_key):
        return (
            agent_request_key in self.goal.request_slots
            and agent_request_key in self.state.rest_slots
        )

    def prepare_dontcare(self, agent_request_key):
        # Fourth and Final Case: otherwise the user sim does not care about the slot being requested, then inform
        # 'anything' as the value of the requested slot
        assert agent_request_key not in self.state.rest_slots
        self.state.intent = "inform"
        self.state.inform_slots[agent_request_key] = "anything"
        self.state.request_slots.clear()
        self.state.history_slots[agent_request_key] = "anything"

    def handle_third_case(self, agent_request_key):
        # Third Case: if the agent requests for something in the user sims goal request slots and it HASN'T been
        # informed, then request it with a random inform
        self.state.request_slots.clear()
        self.state.intent = "request"
        self.state.request_slots[agent_request_key] = "UNK"
        rest_informs = {}
        for key, value in self.state.rest_slots.items():
            if value != "UNK":
                rest_informs[key] = value
        if rest_informs:
            key_choice, value_choice = random.choice(list(rest_informs.items()))
            self.state.inform_slots[key_choice] = value_choice
            self.state.rest_slots.pop(key_choice)
            self.state.history_slots[key_choice] = value_choice

    def inform_agent_about_what_he_should_know(self, agent_request_key):
        # Second Case: if the agent requests for something in user sims goal request slots and it has already been
        # informed, then inform it
        self.state.intent = "inform"
        self.state.inform_slots[agent_request_key] = self.state.history_slots[
            agent_request_key
        ]
        self.state.request_slots.clear()
        assert agent_request_key not in self.state.rest_slots

    def handle_meaningful_agent_request(self, agent_request_key):
        # First Case: if agent requests for something that is in the user sims goal inform slots, then inform it
        self.state.intent = "inform"
        self.state.inform_slots[agent_request_key] = self.goal.inform_slots[
            agent_request_key
        ]
        self.state.request_slots.clear()
        self.state.rest_slots.pop(agent_request_key, None)
        self.state.history_slots[agent_request_key] = self.goal.inform_slots[
            agent_request_key
        ]

    def agent_offers_new_information(self, agent_inform_key, agent_inform_value):
        return agent_inform_value != self.goal.inform_slots.get(
            agent_inform_key, agent_inform_value
        )

    def _response_to_inform(self, agent_action: DialogAction):

        agent_inform_key, agent_inform_value = list(agent_action.inform_slots.items())[
            0
        ]

        assert agent_inform_key != self.default_key

        self.state.history_slots[agent_inform_key] = agent_inform_value
        self.state.rest_slots.pop(agent_inform_key, None)
        self.state.request_slots.pop(agent_inform_key, None)

        if self.agent_offers_new_information(agent_inform_key, agent_inform_value):
            self.handle_the_meaningful_inform(agent_inform_key)
        else:  # Second Case: Otherwise pick a random action to take
            self.handle_meaningless_inform()

    def handle_meaningless_inform(self):
        # - If anything in state requests then request it
        if len(self.state.request_slots) > 0:
            self.state.intent = "request"
        # - Else if something to say in rest slots, pick something
        elif self.state.rest_slots:
            def_in = self.state.rest_slots.pop(self.default_key, False)
            if self.state.rest_slots:
                key, value = random.choice(list(self.state.rest_slots.items()))
                if value != "UNK":
                    self.state.intent = "inform"
                    self.state.inform_slots[key] = value
                    self.state.rest_slots.pop(key)
                    self.state.history_slots[key] = value
                else:
                    self.state.intent = "request"
                    self.state.request_slots[key] = "UNK"
            else:
                self.state.intent = "request"
                self.state.request_slots[self.default_key] = "UNK"
            if def_in == "UNK":
                self.state.rest_slots[self.default_key] = "UNK"
        # - Otherwise respond with 'nothing to say' intent
        else:
            self.state.intent = "thanks"

    def handle_the_meaningful_inform(self, agent_inform_key):
        self.state.intent = "inform"
        self.state.inform_slots[agent_inform_key] = self.goal.inform_slots[
            agent_inform_key
        ]
        self.state.request_slots.clear()
        self.state.history_slots[agent_inform_key] = self.goal.inform_slots[
            agent_inform_key
        ]

    def _response_to_match_found(self, agent_informs: dict):
        self.state.intent = "thanks"
        self.constraint_check = SUCCESS

        assert self.default_key in agent_informs
        self.state.rest_slots.pop(self.default_key, None)
        self.state.history_slots[self.default_key] = str(
            agent_informs[self.default_key]
        )
        self.state.request_slots.pop(self.default_key, None)

        if agent_informs[self.default_key] == "no match available":
            self.constraint_check = FAIL

        self.check_if_matches_goal(agent_informs)

        if self.constraint_check == FAIL:
            self.state.intent = "reject"
            self.state.request_slots.clear()

    def check_if_matches_goal(self, datum: dict):
        slots_that_must_match = list(
            filter(lambda kv: kv[0] not in self.no_query, self.goal.inform_slots.items())
        )

        if any([value != datum.get(key, None) for key, value in slots_that_must_match]):
            self.constraint_check = FAIL

    def _response_to_done(self):

        if self.constraint_check == FAIL:
            return FAIL

        if not self.state.rest_slots:
            assert not self.state.request_slots
        if self.state.rest_slots:
            return FAIL

        return SUCCESS
