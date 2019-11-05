# Special slot values (for reference)
PLACEHOLDER = "PLACEHOLDER"  # For informs
import copy
import json
from dataclasses import dataclass
from typing import Dict

UNK = "UNK"  # For requests
"anything"  # means any value works for the slot with this value
"no match available"  # When the intent of the agent is match_found yet no db match fits current constraints

#######################################
# Usersim Config
#######################################
# Used in EMC for intent error (and in user)
usersim_intents = ["inform", "request", "thanks", "reject", "done"]

# The goal of the agent is to inform a match for this key
usersim_default_key = "ticket"

# Required to be in the first action in inform slots of the usersim if they exist in the goal inform slots
usersim_required_init_inform_keys = ["moviename"]

#######################################
# Agent Config
#######################################
USER = "USER"
AGENT = 'AGENT'

# Possible inform and request slots for the agent
agent_inform_slots = [
    "moviename",
    "theater",
    "starttime",
    "date",
    "genre",
    "state",
    "city",
    "zip",
    "critic_rating",
    "mpaa_rating",
    "distanceconstraints",
    "video_format",
    "theater_chain",
    "price",
    "actor",
    "description",
    "other",
    "numberofkids",
    usersim_default_key,
]
agent_request_slots = [
    "moviename",
    "theater",
    "starttime",
    "date",
    "numberofpeople",
    "genre",
    "state",
    "city",
    "zip",
    "critic_rating",
    "mpaa_rating",
    "distanceconstraints",
    "video_format",
    "theater_chain",
    "price",
    "actor",
    "description",
    "other",
    "numberofkids",
]

@dataclass
class DialogAction: # should be (NamedTuple)
    intent: str
    inform_slots: Dict[str, str] = None
    request_slots: Dict[str, str] = None
    turn:int = 0
    speaker:str= AGENT


inform_actions = [
    DialogAction("inform", {slot: "PLACEHOLDER"})
    for slot in agent_inform_slots
    if slot != usersim_default_key
]
request_actions = [
    DialogAction("request", request_slots={slot: "UNK"})
    for slot in agent_request_slots
]

AGENT_ACTIONS = (
        [DialogAction("done"), DialogAction("match_found")] + inform_actions + request_actions
)


# Rule-based policy request list
RULE_REQUESTS = ["moviename", "starttime", "city", "date", "theater", "numberofpeople"]

# These are possible inform slot keys that cannot be used to query
no_query_keys = ["numberofpeople", usersim_default_key]

#######################################
# Global config
#######################################

# These are used for both constraint check AND success check in usersim
FAIL = -1
NO_OUTCOME = 0
SUCCESS = 1

# All possible intents (for one-hot conversion in ST.get_state())
all_intents = ["inform", "request", "done", "match_found", "thanks", "reject"]

# All possible slots (for one-hot conversion in ST.get_state())
all_slots = [
    "actor",
    "actress",
    "city",
    "critic_rating",
    "date",
    "description",
    "distanceconstraints",
    "genre",
    "greeting",
    "implicit_value",
    "movie_series",
    "moviename",
    "mpaa_rating",
    "numberofpeople",
    "numberofkids",
    "other",
    "price",
    "seating",
    "starttime",
    "state",
    "theater",
    "theater_chain",
    "video_format",
    "zip",
    "result",
    usersim_default_key,
    "mc_list",
]

idx2action={i:a for i,a in enumerate(AGENT_ACTIONS)}
action2idx = {json.dumps(v.__dict__):k for k,v in idx2action.items()}

def map_index_to_action(index):
    return copy.deepcopy(idx2action[index])

def map_action_to_index(response):
    for (i, action) in enumerate(AGENT_ACTIONS):
        if response == action:
            return i
    raise ValueError("Response: {} not found in possible actions".format(response))


# def map_action_to_index(response):
#     return action2idx[json.dumps(response)]