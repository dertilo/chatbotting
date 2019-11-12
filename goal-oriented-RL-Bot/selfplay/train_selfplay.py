import json

from dialog_agent_env import load_data



if __name__ == "__main__":

    DATABASE_FILE_PATH = "../data/movie_db.pkl"
    DICT_FILE_PATH = "../data/movie_dict.pkl"
    USER_GOALS_FILE_PATH = "../data/movie_user_goals.pkl"

    train_params = {
        "num_ep_run": 40000,
        "train_freq": 100,
        "max_round_num": 20,
    }

    slot2values, database, user_goals = load_data(
        DATABASE_FILE_PATH, DICT_FILE_PATH, USER_GOALS_FILE_PATH
    )

    print()
