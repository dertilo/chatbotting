from Experience import Experience
from error_model_controller import ErrorModelController
from state_tracker import StateTracker
from user_simulator import UserSimulator


class DialogEnv(object):
    def __init__(
        self,
        user: UserSimulator,
        emc: ErrorModelController,
        state_tracker: StateTracker,
    ) -> None:
        self.user = user
        self.emc = emc
        self.state_tracker = state_tracker

    def step(self, agent_action):
        self.state_tracker.update_state_agent(agent_action)
        user_action, reward, done, success = self.user.step(agent_action)
        if not done:
            self.emc.infuse_error(user_action)
        self.state_tracker.update_state_user(user_action)
        next_state = self.state_tracker.get_state(done)
        return next_state, reward, done, success

    def reset(self):
        self.state_tracker.reset()
        init_user_action = self.user.reset()
        self.emc.infuse_error(init_user_action)
        self.state_tracker.update_state_user(init_user_action)
        return self.state_tracker.get_state()


def run_dialog_episode(
    agent, dialog_env: DialogEnv, experience: Experience, num_max_steps=30
):
    state = dialog_env.reset()
    turn = 0
    reward_sum = 0
    for turn in range(1, num_max_steps + 1):
        agent_action_index, agent_action = agent.get_action(state)
        next_state, reward, done, success = dialog_env.step(agent_action)

        experience.add_experience(state,agent_action_index,next_state,reward,done)

        state = next_state
        reward_sum += reward
        if done:
            break
    return turn, reward_sum, success
