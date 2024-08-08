import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def save_frame(
    folder: str,
    timestamp: int,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # add action to obs

    # make folder if it doesn't exist
    # folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder + str(timestamp) + ".pkl"
    print(recorded_file)

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

def save_action(recorded_file,action: np.ndarray):
    with open(recorded_file, "ab") as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    # test write
    act = [1,2,3,4,5,6]



