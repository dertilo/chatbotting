from tqdm import tqdm

from Experience import Experience
from rulebased_agent import RuleBasedAgent
from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
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


def one_round_agent_user_action_collect_experience(
    agent, experience: Experience, state_tracker: StateTracker, state
):
    agent_action_index, agent_action = agent.get_action(state)

    state_tracker.update_state_agent(agent_action)
    user_action, reward, done, success = user.step(agent_action)
    if not done:
        emc.infuse_error(user_action)

    state_tracker.update_state_user(user_action)
    next_state = state_tracker.get_state(done)
    experience.add_experience(state, agent_action_index, reward, next_state, done)

    return next_state, reward, done, success


def warmup_run(
    agent: RuleBasedAgent,
    experience: Experience,
    state_tracker: StateTracker,
    num_warmup_steps: int,
):

    print("Warmup Started...")
    total_step = 0
    while total_step != num_warmup_steps and not experience.is_memory_full():
        agent.reset_rulebased_vars()
        num_steps, _, _ = run_dialog_episode(agent, experience, state_tracker)
        total_step += num_steps

    print("...Warmup Ended")


def run_dialog_episode(agent, experience: Experience, state_tracker, num_max_steps=30):
    episode_reset(state_tracker, user, emc)
    state = state_tracker.get_state()
    turn = 0
    reward_sum = 0
    for turn in range(1, num_max_steps + 1):
        next_state, reward, done, success = one_round_agent_user_action_collect_experience(
            agent, experience, state_tracker, state
        )
        state = next_state
        reward_sum += reward
        if done:
            break
    return turn, reward_sum, success


def run_train(train_params):

    NUM_EP_TRAIN = train_params["num_ep_run"]
    TRAIN_INTERVAL = train_params["train_freq"]

    print("Training Started...")
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0

    params_to_monitor = {"dialogue": 0, "success-rate": 0.0, "avg-reward": 0.0}
    running_factor = 0.9
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialog_counter in range(NUM_EP_TRAIN):

            num_turns, dialog_reward, success = run_dialog_episode(
                dqn_agent, experience, state_tracker
            )
            period_reward_total += dialog_reward
            period_success_total += int(success)

            if dialog_counter % TRAIN_INTERVAL == 0:
                success_rate = period_success_total / TRAIN_INTERVAL
                avg_reward = period_reward_total / TRAIN_INTERVAL

                flushed_agent_memory, success_rate_best = handle_successfulness(
                    success_rate, avg_reward,
                    dqn_agent,experience,
                    dialog_counter, success_rate_best
                )
                period_success_total = 0
                period_reward_total = 0
                dqn_agent.update_target_model_weights()
                if not flushed_agent_memory:
                    dqn_agent.train(experience)

                update_progess_bar(
                    pbar, dialog_counter, avg_reward, running_factor, success_rate
                )

    print("...Training Ended")


def update_progess_bar(pbar, dialog_counter, avg_reward, running_factor, success_rate):
    pbar.postfix[0]["dialogue"] = dialog_counter
    pbar.postfix[0]["success-rate"] = round(
        running_factor * pbar.postfix[0]["success-rate"]
        + (1 - running_factor) * success_rate,
        2,
    )
    pbar.postfix[0]["avg-reward"] = round(
        running_factor * pbar.postfix[0]["avg-reward"]
        + (1 - running_factor) * avg_reward,
        2,
    )
    pbar.update()


def handle_successfulness(success_rate, avg_reward,
                          agent, experience:Experience,
                          episode_counter,
                          success_rate_best):
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
    flushed_agent_memory = (
            is_new_highscore and success_rate >= SUCCESS_RATE_THRESHOLD
    )
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


def episode_reset(
    state_tracker: StateTracker, user: UserSimulator, emc: ErrorModelController
):
    state_tracker.reset()
    init_user_action = user.reset()
    emc.infuse_error(init_user_action)
    state_tracker.update_state_user(init_user_action)


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
    warmup_run(rule_agent, experience, state_tracker, train_params["warmup_mem"])
    run_train(train_params)
