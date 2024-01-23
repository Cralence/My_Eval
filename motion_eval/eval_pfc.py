import argparse
import glob
import os
import pickle
import random
import numpy as np
from tqdm import tqdm


def calc_physical_score(dir, fps):
    scores = []
    names = []
    accelerations = []
    up_dir = 1  # z is up
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / fps

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        # joint3d = np.load(pkl)
        joint3d = info["keypoints3d"]
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        print(f'{pkl}, joint: {joint3d.shape}, root_v: {root_v.shape}, root_a: {root_a.shape}', end=' ')
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)
        scaling = root_a.max()
        root_a /= scaling

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        )  # (S-2, 4) horizontal velocity
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1])
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3])

        print(f'foot_v: {foot_v.shape}, foot_mins: {foot_mins.shape}')

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"{dir} has a mean PFC of {out}")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="../test_aist_joint",
        help="Where to load saved motions",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        choices=[20, 60],
        help="fps of the tested motion",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path, opt.fps)