import json
from time import time

from tqdm import tqdm

from Experience import Experience
from dialog_agent_env import DialogEnv, load_data
from dialogue_config import map_index_to_action
from original_keras_dqn_agent import DQNAgent
from rulebased_agent import RuleBasedAgent

exp_iter = None


def run_dialog_episode(
    agent: DQNAgent, dialog_env: DialogEnv, experience: Experience, num_max_steps=30
):
    state = dialog_env.reset()
    turn = 0
    reward_sum = 0
    for turn in range(1, num_max_steps + 1):
        agent_action_index = agent.step(state)
        agent_action = map_index_to_action(agent_action_index)
        next_state, reward, done, success = dialog_env.step(agent_action)

        experience.add_experience(state, agent_action_index, next_state, reward, done)

        state = next_state
        reward_sum += reward
        if done:
            break
    return turn, reward_sum, success


def warmup_run(
    agent: RuleBasedAgent,
    dialog_env: DialogEnv,
    experience: Experience,
    num_warmup_steps: int,
):

    total_step = 0
    while total_step != num_warmup_steps and not experience.is_memory_full():
        agent.reset()
        num_steps, _, _ = run_dialog_episode(agent, dialog_env, experience)
        total_step += num_steps


def run_train(
    dqn_agent: DQNAgent,
    dialog_env: DialogEnv,
    experience: Experience,
    num_episodes,
    train_freq,
):

    params_to_monitor = {"dialogue": 0, "dialog_reward": 0.0}
    running_factor = 0.9
    reward_sum = 0
    with tqdm(postfix=[params_to_monitor]) as pbar:

        for dialog_counter in range(1, num_episodes + 1):

            num_turns, dialog_reward, success = run_dialog_episode(
                dqn_agent, dialog_env, experience
            )
            reward_sum += dialog_reward
            if dialog_counter % train_freq == 0:

                dqn_agent.update_target_model_weights()
                dqn_agent.train(experience)

                update_progess_bar(
                    pbar, dialog_counter, reward_sum / dialog_counter, running_factor
                )


def update_progess_bar(pbar, dialog_counter, dialog_reward, running_factor):
    pbar.postfix[0]["dialogue"] = dialog_counter
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
    with open("constants.json") as f:
        params = json.load(f)

    file_path_dict = params["db_file_paths"]
    DATABASE_FILE_PATH = file_path_dict["database"]
    DICT_FILE_PATH = file_path_dict["dict"]
    USER_GOALS_FILE_PATH = file_path_dict["user_goals"]
    train_params = params["run"]

    slot2values, database, user_goals = load_data(
        DATABASE_FILE_PATH, DICT_FILE_PATH, USER_GOALS_FILE_PATH
    )

    dialog_env = DialogEnv(
        user_goals, params["emc"], params["run"]["max_round_num"], database, slot2values
    )
    dqn_agent = DQNAgent(dialog_env.state_tracker.get_state_size(), params)
    rule_agent = RuleBasedAgent(params["agent"]["epsilon_init"])
    experience = Experience(100_000)

    SUCCESS_RATE_THRESHOLD = train_params["success_rate_threshold"]
    start = time()
    warmup_run(rule_agent, dialog_env, experience, 10_000)
    # num_episodes = train_params["num_ep_run"]
    num_episodes = 4000
    run_train(
        dqn_agent, dialog_env, experience, num_episodes, train_params["train_freq"]
    )
    print(time() - start)
