from nle.env import tasks


class NetHackStaircase(tasks.NetHackStaircase):
    """Override the base class to set the reward for reaching the staircase."""

    def __init__(
        self, *args, reward_win: float = 1.0, reward_lose: float = 1.0, **kwargs
    ):
        self.reward_win = reward_win
        self.reward_lose = reward_lose
        super().__init__(*args, **kwargs)

    def _reward_fn(self, last_observation, action, observation, end_status):
        del action  # Unused
        time_penalty = self._get_time_penalty(last_observation, observation)
        if end_status == self.StepStatus.TASK_SUCCESSFUL:
            reward = self.reward_win
        else:
            reward = self.reward_lose
        return reward + time_penalty
