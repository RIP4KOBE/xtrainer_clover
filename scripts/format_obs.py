import datetime
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import cv2



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

def save_dp_frame(
    folder: str,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    activated=True,
    save_png=False,
) -> None:
    obs["activated"] = {}
    obs["activated"]["l"]  = activated
    obs["activated"]["r"] = activated
    obs["control"] = action  # add action to obs

    # recorded_file = folder / (
    #     timestamp.isoformat().replace(":", "-").replace(".", "-") + ".pkl"
    # # )
    save_time = timestamp.isoformat().replace(":", "-").replace(".", "-")
    recorded_file = folder + f"{save_time}" + ".pkl"


    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

    # save rgb image as png
    if save_png:
        rgb = obs["base_rgb"]
        for i in range(rgb.shape[0]):
            normalized_img = cv2.normalize(rgb[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            uint_img = normalized_img.astype(np.uint8)
            rgbi = cv2.cvtColor(uint_img, cv2.COLOR_RGB2BGR)
            fn = str(recorded_file)[:-4] + f"-{i}.png"
            cv2.imwrite(fn, rgbi)

def save_action(recorded_file,action: np.ndarray):
    with open(recorded_file, "ab") as f:
        pickle.dump(action, f)


if __name__ == "__main__":
    # test write
    act = [1,2,3,4,5,6]



