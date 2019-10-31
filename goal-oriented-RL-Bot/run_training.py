from user_simulator import UserSimulator
from error_model_controller import ErrorModelController
from dqn_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
from user import User


def get_params(params_json_file="constants.json"):
    global DATABASE_FILE_PATH, DICT_FILE_PATH, USER_GOALS_FILE_PATH, USE_USERSIM, NUM_EP_TRAIN, TRAIN_FREQ, SUCCESS_RATE_THRESHOLD

    with open(params_json_file) as f:
        constants = json.load(f)

    file_path_dict = constants["db_file_paths"]
    DATABASE_FILE_PATH = file_path_dict["database"]
    DICT_FILE_PATH = file_path_dict["dict"]
    USER_GOALS_FILE_PATH = file_path_dict["user_goals"]
    # Load run constants
    run_dict = constants["run"]
    USE_USERSIM = run_dict["usersim"]
    NUM_EP_TRAIN = run_dict["num_ep_run"]
    TRAIN_FREQ = run_dict["train_freq"]
    MAX_ROUND_NUM = run_dict["max_round_num"]
    SUCCESS_RATE_THRESHOLD = run_dict["success_rate_threshold"]
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
        num_steps = run_dialog_episode(dqn_agent, state_tracker)
        total_step += num_steps

    print("...Warmup Ended")


def run_dialog_episode(dqn_agent, state_tracker, num_max_steps=30):
    episode_reset(state_tracker, user, emc, dqn_agent)
    state = state_tracker.get_state()
    step = 0
    for step in range(1, num_max_steps + 1):
        next_state, _, done, _ = one_round_agent_user_action_collect_experience(
            dqn_agent, state_tracker, state, warmup=True
        )
        state = next_state
        if done:
            break
    return step


def train_run():
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """

    print("Training Started...")
    episode = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    while episode < NUM_EP_TRAIN:
        episode_reset(state_tracker, user, emc, dqn_agent)
        episode += 1
        done = False
        state = state_tracker.get_state()
        while not done:
            next_state, reward, done, success = one_round_agent_user_action_collect_experience(
                dqn_agent,state_tracker,state
            )
            period_reward_total += reward
            state = next_state

        period_success_total += success

        # Train
        if episode % TRAIN_FREQ == 0:
            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            avg_reward = period_reward_total / TRAIN_FREQ
            # Flush
            if (
                success_rate >= success_rate_best
                and success_rate >= SUCCESS_RATE_THRESHOLD
            ):
                dqn_agent.empty_memory()
            # Update current best success rate
            if success_rate > success_rate_best:
                print(
                    "Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}".format(
                        episode, success_rate, avg_reward
                    )
                )
                success_rate_best = success_rate
                dqn_agent.save_weights()
            period_success_total = 0
            period_reward_total = 0
            # Copy
            dqn_agent.copy()
            # Train
            dqn_agent.train()
    print("...Training Ended")


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
    train_run()
