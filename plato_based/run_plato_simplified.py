from pprint import pprint
from tqdm import tqdm

from ConversationalSingleAgentSimplified import ConversationalSingleAgent

import yaml
import os.path
import time
import random


def run_single_agent(config, num_dialogues):
    ca = ConversationalSingleAgent(config)
    ca.initialize()

    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "reward": 0.0}
    running_factor = 0.99
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialogue in range(num_dialogues):
            ca.start_dialogue()
            while not ca.terminated():
                ca.continue_dialogue()

            ca.end_dialogue()

            update_progress_bar(ca, dialogue, pbar, running_factor)

    statistics = collect_statistics(ca, num_dialogues)

    print(
        "\n\nDialogue Success Rate: {0}\nAverage Cumulative Reward: {1}"
        "\nAverage Turns: {2}".format(
            statistics["AGENT_0"]["dialogue_success_percentage"],
            statistics["AGENT_0"]["avg_cumulative_rewards"],
            statistics["AGENT_0"]["avg_turns"],
        )
    )

    return statistics


def collect_statistics(ca, num_dialogues):
    statistics = {"AGENT_0": {}}
    statistics["AGENT_0"]["dialogue_success_percentage"] = 100 * float(
        ca.num_successful_dialogues / num_dialogues
    )
    statistics["AGENT_0"]["avg_cumulative_rewards"] = float(
        ca.cumulative_rewards / num_dialogues
    )
    statistics["AGENT_0"]["avg_turns"] = float(ca.total_dialogue_turns / num_dialogues)
    statistics["AGENT_0"]["objective_task_completion_percentage"] = 100 * float(
        ca.num_task_success / num_dialogues
    )
    return statistics


def update_progress_bar(ca, dialogue, pbar, running_factor):
    pbar.postfix[0]["dialogue"] = dialogue
    success = int(ca.recorder.dialogues[-1][-1].success)
    reward = int(ca.recorder.dialogues[-1][-1].cumulative_reward)
    eps = ca.dialogue_manager.policy.epsilon
    pbar.postfix[0]["eps"] = eps

    pbar.postfix[0]["success-rate"] = round(
        running_factor * pbar.postfix[0]["success-rate"]
        + (1 - running_factor) * success,
        2,
    )
    pbar.postfix[0]["reward"] = round(
        running_factor * pbar.postfix[0]["reward"] + (1 - running_factor) * reward, 2
    )
    pbar.update()



if __name__ == "__main__":
    config = {
        "GENERAL": {
            "agents": 1,
            "runs": 5,
            "experience_logs": {
                "save": True,
                "load": False,
                "path": "Logs/simulate_agenda",
            },
        },
        "DIALOGUE": {
            "num_dialogues": 1000,
            "initiative": "system",
            # "domain": "CamRest",
            "ontology_path": "/home/tilo/code/OKS/alex-plato/Domain/alex-rules.json",
            "db_path": "/home/tilo/code/OKS/alex-plato/Domain/alex-dbase.db",
            "db_type": "sql",
        },
        "AGENT_0": {
            # "role": "system",
            # "USER_SIMULATOR": {
            #     "simulator": "agenda",
            #     "patience": 5,
            #     "pop_distribution": [1.0],
            #     "slot_confuse_prob": 0.0,
            #     "op_confuse_prob": 0.0,
            #     "value_confuse_prob": 0.0,
            # },
            "DM": {
                "policy": {
                    "type": "reinforce",
                    "train": True,
                    "learning_rate": 0.25,
                    "exploration_rate": 1.0,
                    "discount_factor": 0.95,
                    "learning_decay_rate": 0.95,
                    "exploration_decay_rate": .95,
                    "policy_path": "/tmp/policy_sys.pkl",
                }
            },
            "DST": {"dst": "dummy"},
        },
    }

    statistics = run_single_agent(config, 400)

    pprint(f"Results:\n{statistics}")
