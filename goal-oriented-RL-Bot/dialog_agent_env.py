from Experience import Experience
from error_model_controller import ErrorModelController
from state_tracker import StateTracker
from user_simulator import UserSimulator


def episode_reset(
    state_tracker: StateTracker, user: UserSimulator, emc: ErrorModelController
):
    state_tracker.reset()
    init_user_action = user.reset()
    emc.infuse_error(init_user_action)
    state_tracker.update_state_user(init_user_action)


def one_round_agent_user_action(agent, user, emc, state_tracker: StateTracker, state):
    agent_action_index, agent_action = agent.get_action(state)

    state_tracker.update_state_agent(agent_action)
    user_action, reward, done, success = user.step(agent_action)
    if not done:
        emc.infuse_error(user_action)

    state_tracker.update_state_user(user_action)
    next_state = state_tracker.get_state(done)

    return state, agent_action_index, next_state, reward, done, success


def run_dialog_episode(
    agent,
    user: UserSimulator,
    emc: ErrorModelController,
    experience: Experience,
    state_tracker,
    num_max_steps=30,
):
    episode_reset(state_tracker, user, emc)
    state = state_tracker.get_state()
    turn = 0
    reward_sum = 0
    for turn in range(1, num_max_steps + 1):
        experience_atom = one_round_agent_user_action(
            agent, user, emc, state_tracker, state
        )
        state, agent_action_index, next_state, reward, done, success = experience_atom
        experience.add_experience(*experience_atom[:-1])

        state = next_state
        reward_sum += reward
        if done:
            break
    return turn, reward_sum, success
