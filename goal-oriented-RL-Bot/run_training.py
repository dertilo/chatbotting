from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User


def get_params(params_json_file="constants.json"):
    global DATABASE_FILE_PATH, DICT_FILE_PATH, USER_GOALS_FILE_PATH, USE_USERSIM, NUM_EP_TRAIN, TRAIN_INTERVAL, SUCCESS_RATE_THRESHOLD

    with open(params_json_file) as f:
        constants = json.load(f)

    file_path_dict = constants["db_file_paths"]
    DATABASE_FILE_PATH = file_path_dict["database"]
    DICT_FILE_PATH = file_path_dict["dict"]
    USER_GOALS_FILE_PATH = file_path_dict["user_goals"]
    # Load run constants
    return constants


def one_round_agent_user_action_collect_experience(
    dqn_agent: DQNAgent, state_tracker: StateTracker, state, warmup=False
):

    agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=warmup)
    state_tracker.update_state_agent(agent_action)
    user_action, reward, done, success = user.step(agent_action)
    if not done:
        emc.infuse_error(user_action)

    state_tracker.update_state_user(user_action)
    next_state = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

    return next_state, reward, done, success


def warmup_run(dqn_agent: DQNAgent, state_tracker: StateTracker, num_warmup_steps: int):

    print("Warmup Started...")
    total_step = 0
    while total_step != num_warmup_steps and not dqn_agent.is_memory_full():
        num_steps, _, _ = run_dialog_episode(dqn_agent, state_tracker)
        total_step += num_steps

    print("...Warmup Ended")


def run_dialog_episode(dqn_agent, state_tracker, num_max_steps=30):
    episode_reset(state_tracker, user, emc, dqn_agent)
    state = state_tracker.get_state()
    turn = 0
    reward_sum = 0
    for turn in range(1, num_max_steps + 1):
        next_state, reward, done, success = one_round_agent_user_action_collect_experience(
            dqn_agent, state_tracker, state, warmup=True
        )
        state = next_state
        reward_sum += reward
        if done:
            break
    return turn, reward_sum, success


def run_train(train_params):

    NUM_EP_TRAIN = train_params["num_ep_run"]
    TRAIN_INTERVAL = train_params["train_freq"]
    SUCCESS_RATE_THRESHOLD = train_params["success_rate_threshold"]

    print("Training Started...")
    episode_counter = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    while episode_counter < NUM_EP_TRAIN:
        episode_counter += 1

        num_turns, dialog_reward, success = run_dialog_episode(dqn_agent, state_tracker)
        period_reward_total += dialog_reward
        period_success_total += int(success)

        if episode_counter % TRAIN_INTERVAL == 0:
            success_rate_best = handle_successfulness(
                SUCCESS_RATE_THRESHOLD,
                TRAIN_INTERVAL,
                episode_counter,
                period_reward_total,
                period_success_total,
                success_rate_best,
            )
            period_success_total = 0
            period_reward_total = 0
            dqn_agent.copy() #TODO(tilo): why?
            dqn_agent.train()
    print("...Training Ended")


def handle_successfulness(
    SUCCESS_RATE_THRESHOLD,
    TRAIN_INTERVAL,
    episode_counter,
    period_reward_total,
    period_success_total,
    success_rate_best,
):
    success_rate = period_success_total / TRAIN_INTERVAL
    avg_reward = period_reward_total / TRAIN_INTERVAL
    if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
        dqn_agent.empty_memory()

    # Update current best success rate
    if success_rate > success_rate_best:
        print(
            "Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}".format(
                episode_counter, success_rate, avg_reward
            )
        )
        success_rate_best = success_rate
        dqn_agent.save_weights()
    return success_rate_best


def episode_reset(
    state_tracker: StateTracker,
    user: UserSimulator,
    emc: ErrorModelController,
    dqn_agent: DQNAgent,
):
    state_tracker.reset()
    init_user_action = user.reset()
    emc.infuse_error(init_user_action)
    state_tracker.update_state_user(init_user_action)
    dqn_agent.reset()


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

    warmup_run(dqn_agent, state_tracker, train_params["warmup_mem"])
    run_train(train_params)
