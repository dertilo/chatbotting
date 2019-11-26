import Ontology
from State import SlotFillingDialogueState
from DialoguePolicy import DialoguePolicy
from dialog_action_classes import DialogueAct, DialogueActItem, Operator

from copy import deepcopy

import random

"""
HandcraftedPolicy is a rule-based system policy, developed as a baseline and as
a quick way to perform sanity checks and debug a Conversational Agent. 
It will try to fill unfilled slots, then suggest an item, and answer any 
requests from the user.
"""


def get_value(item_in_focus, requested_slot):
    if requested_slot in item_in_focus and item_in_focus[requested_slot]:
        value = item_in_focus[requested_slot]
    else:
        value = "not available"
    return value


def build_inform_act(dialogue_state: SlotFillingDialogueState):
    requested_slot = dialogue_state.requested_slot
    # Reset request as we attempt to address it
    dialogue_state.requested_slot = ""
    value = get_value(dialogue_state.item_in_focus, requested_slot)
    dact = [
        DialogueAct("inform", [DialogueActItem(requested_slot, Operator.EQ, value)])
    ]
    return dact


def make_request(unfilled_slots):
    slot = random.choice(unfilled_slots)
    dacts = [DialogueAct("request", [DialogueActItem(slot, Operator.EQ, "")])]
    return dacts


def make_offer(ds):
    name = ds.item_in_focus["name"] if "name" in ds.item_in_focus else "unknown"
    dacts = [DialogueAct("offer", [DialogueActItem("name", Operator.EQ, name)])]
    inform_acts = [
        build_inform(slot, ds)
        for slot in ds.slots_filled
        if slot != "requested" and ds.slots_filled[slot] and slot not in ["id", "name"]
    ]
    dacts += inform_acts
    return dacts


def build_inform(slot, ds: SlotFillingDialogueState):
    if slot in ds.item_in_focus:
        value = ds.item_in_focus[slot]
    else:
        value = "no info"

    return DialogueAct("inform", [DialogueActItem(slot, Operator.EQ, value)])


class HandcraftedPolicy:
    def __init__(self, ontology: Ontology.Ontology):
        super(HandcraftedPolicy, self).__init__()
        self.ontology = ontology

    def next_action(self, ds: SlotFillingDialogueState):
        if ds.is_terminal_state:
            dacts = [DialogueAct("bye", [DialogueActItem("", Operator.EQ, "")])]
        elif ds.requested_slot != "" and ds.item_in_focus and ds.system_made_offer:
            dacts = build_inform_act(ds)
        else:
            dacts = self.request_slots_or_make_offer(ds)
        return dacts

    def request_slots_or_make_offer(self, ds: SlotFillingDialogueState):
        unfilled_slots = [
            s
            for s in self.ontology.ontology["system_requestable"]
            if ds.slots_filled[s] is None
        ]
        if len(unfilled_slots) > 0:
            dacts = make_request(unfilled_slots)
        elif ds.item_in_focus:
            dacts = make_offer(ds)
        else:
            dacts = [DialogueAct("canthelp", [])]
        return dacts
