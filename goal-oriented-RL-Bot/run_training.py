from tqdm import tqdm

from Experience import Experience
from dialog_agent_env import run_dialog_episode, DialogEnv
from rulebased_agent import RuleBasedAgent
from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, json
from utils import remove_empty_slots


def get_params(params_json_file="constants.json"):
    global DATABASE_FILE_PATH, DICT_FILE_PATH, USER_GOALS_FILE_PATH, USE_USERSIM, NUM_EP_TRAIN, TRAIN_INTERVAL

    with open(params_json_file) as f:
        constants = json.load(f)

    file_path_dict = constants["db_file_paths"]
    DATABASE_FILE_PATH = file_path_dict["database"]
    DICT_FILE_PATH = file_path_dict["dict"]
    USER_GOALS_FILE_PATH = file_path_dict["user_goals"]
    return constants


def warmup_run(
    agent: RuleBasedAgent,
    dialog_env: DialogEnv,
    experience: Experience,
    num_warmup_steps: int,
):

    total_step = 0
    while total_step != num_warmup_steps and not experience.is_memory_full():
        agent.reset_rulebased_vars()
        num_steps, _, _ = run_dialog_episode(agent, dialog_env, experience)
        total_step += num_steps


def run_train(user_env: DialogEnv, train_params):

    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "dialog_reward": 0.0}
    running_factor = 0.9
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialog_counter in range(train_params["num_ep_run"]):

            num_turns, dialog_reward, success = run_dialog_episode(
                dqn_agent, user_env, experience
            )

            if dialog_counter % train_params["train_freq"] == 0:

                dqn_agent.update_target_model_weights()
                dqn_agent.train(experience)

                update_progess_bar(
                    pbar, dialog_counter, dialog_reward, running_factor, int(success)
                )

    print("...Training Ended")


def update_progess_bar(
    pbar, dialog_counter, dialog_reward, running_factor, success_rate
):
    pbar.postfix[0]["dialogue"] = dialog_counter
    pbar.postfix[0]["success-rate"] = round(
        running_factor * pbar.postfix[0]["success-rate"]
        + (1 - running_factor) * success_rate,
        2,
    )
    pbar.postfix[0]["dialog_reward"] = round(
        running_factor * pbar.postfix[0]["dialog_reward"]
        + (1 - running_factor) * dialog_reward,
        2,
    )
    pbar.update()


def handle_successfulness(
    success_rate,
    avg_reward,
    agent,
    experience: Experience,
    episode_counter,
    success_rate_best,
):
    is_new_highscore = success_rate > success_rate_best
    # flushed_agent_memory = False
    """
    If the success rate of that period is greater than or equal to the 
    current best success rate (initialized to 0.0 at the start of 
    train_run()) AND it is higher than some SUCCESS_RATE_THRESHOLD, 
    then the agent’s memory is emptied. This is to get rid of older 
    experiences that are based on actions of the previous version of the agent’s 
    model i.e. actions that were taken by a less optimal model. 
    This then allows newer experiences from the better version of the model to 
    fill the memory. This way the training and performance is stabilized.
    """
    flushed_agent_memory = is_new_highscore and success_rate >= SUCCESS_RATE_THRESHOLD
    if flushed_agent_memory:
        experience.empty_memory()

    if is_new_highscore:
        print(
            "Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}".format(
                episode_counter, success_rate, avg_reward
            )
        )
        success_rate_best = success_rate
        agent.save_weights()
    return flushed_agent_memory, success_rate_best


if __name__ == "__main__":
    params = get_params()
    train_params = params["run"]

    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, "rb"), encoding="latin1")
    remove_empty_slots(database)

    db_dict = pickle.load(open(DICT_FILE_PATH, "rb"), encoding="latin1")
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, "rb"), encoding="latin1")
    user = UserSimulator(user_goals, params, database)

    emc = ErrorModelController(db_dict, params)
    state_tracker = StateTracker(database, params)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), params)
    rule_agent = RuleBasedAgent(params["agent"]["epsilon_init"])
    experience = Experience(params["agent"]["max_mem_size"])

    SUCCESS_RATE_THRESHOLD = train_params["success_rate_threshold"]
    dialog_env = DialogEnv(user, emc, state_tracker)
    warmup_run(rule_agent, dialog_env, experience, train_params["warmup_mem"])
    run_train(dialog_env, train_params)
