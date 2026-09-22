"""Recording the multitask dataset, and loading it as action-chunk windows.

Recording: every training (task, colour) combination from the bench, each
with `distractors=True` so all three cubes are on the table and the colour
word in the instruction is the only thing that says which one. Successful
scripted-expert episodes only. Per frame: the 15-d proprio state, the front
camera image, the expert's absolute joint targets as the action, and one
instruction sampled from the TRAINING templates. Written as a LeRobotDataset
v3 through the bench's recorder.

Loading: the dataset is read once into flat numpy arrays (uint8 images,
float32 state/action, one instruction string and episode id per frame) and
cached as `.npz` next to it. Windows are (frame t, actions t..t+k-1), padded
with the episode's last action and masked. The split is by episode, so no
frame of a validation episode is ever a training window's neighbour.
Normalisation statistics are computed on training frames only.
"""
import json
import os
import time

import numpy as np

from so_arm100_sim import instructions as I
from so_arm100_sim.env import ArmEnv, CONTROL_HZ, PROPRIO_DIM, ACT_DIM
from so_arm100_sim.expert import ScriptedExpert

IMAGE = dict(camera="front", height=96, width=128)


# -- recording -----------------------------------------------------------

def record_episode(env, expert, instruction, seed):
    obs = env.reset(seed=seed)
    expert.reset()
    frames = []
    while True:
        a = expert.act()
        frames.append({"state": obs["state"], "action": a, "task": instruction,
                       "images": {"front": obs["image"]}})
        obs, _, done, info = env.step(a)
        if done:
            return frames, bool(info["success_ever"])


def record_multitask(root, repo_id, episodes_per_combo=20, combos=None, seed=0,
                     height=IMAGE["height"], width=IMAGE["width"], max_attempts=3,
                     dr=None, log=print):
    """Successful expert demos for every combo -> LeRobotDataset at `root`."""
    from so_arm100_sim.datasets import LeRobotRecorder
    combos = list(I.TRAIN_COMBOS if combos is None else combos)
    rec = LeRobotRecorder(root, repo_id, fps=CONTROL_HZ, cameras=("front",),
                          height=height, width=width)
    rng = np.random.default_rng(seed)
    summary = {"combos": {}, "height": height, "width": width, "seed": seed}
    t0 = time.perf_counter()
    for k, (task, color) in enumerate(combos):
        env = ArmEnv(f"{task}:{color}:distractors", obs_mode="proprio", action_mode="absolute",
                     seed=seed * 1000 + k, dr=dr,
                     image={"camera": "front", "height": height, "width": width})
        ex = ScriptedExpert(env)
        kept, attempts, n_frames = 0, 0, 0
        while kept < episodes_per_combo and attempts < max_attempts * episodes_per_combo:
            instr = I.sample_instruction(task, color, rng, "train")
            frames, ok = record_episode(env, ex, instr, seed=seed * 1_000_000 + k * 10_000 + attempts)
            attempts += 1
            if ok:
                rec.add_episode(frames)
                kept += 1
                n_frames += len(frames)
        env.close()
        summary["combos"][f"{task}:{color}"] = {"kept": kept, "attempts": attempts,
                                                 "frames": n_frames}
        log(f"  {task}:{color}  kept {kept}/{attempts}  frames {n_frames}  "
            f"{time.perf_counter() - t0:.0f} s")
    rec.finalize()
    summary["episodes"], summary["frames"] = rec.n_episodes, rec.n_frames
    summary["wall_s"] = time.perf_counter() - t0
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "recording_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# -- loading -------------------------------------------------------------

def _to_uint8_hwc(img):
    import torch
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img


def load_flat(root, repo_id, cache=True):
    """LeRobotDataset -> dict of flat arrays, cached as flat.npz under root."""
    cache_path = os.path.join(root, "flat.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from so_arm100_sim.datasets import load
    ds = load(root, repo_id)
    n = len(ds)
    states = np.zeros((n, PROPRIO_DIM), np.float32)
    actions = np.zeros((n, ACT_DIM), np.float32)
    episode = np.zeros(n, np.int64)
    frame = np.zeros(n, np.int64)
    tasks = []
    images = None
    for i in range(n):
        item = ds[i]
        states[i] = np.asarray(item["observation.state"], np.float32)
        actions[i] = np.asarray(item["action"], np.float32)
        episode[i] = int(item["episode_index"])
        frame[i] = int(item["frame_index"])
        tasks.append(str(item["task"]))
        img = _to_uint8_hwc(item["observation.images.front"])
        if images is None:
            images = np.zeros((n,) + img.shape, np.uint8)
        images[i] = img
    flat = {"states": states, "actions": actions, "episode": episode, "frame": frame,
            "images": images, "tasks": np.array(tasks)}
    if cache:
        np.savez(cache_path, **flat)
    return flat


class Stats:
    def __init__(self, state_mean, state_std, action_mean, action_std):
        self.state_mean = np.asarray(state_mean, np.float32)
        self.state_std = np.asarray(state_std, np.float32)
        self.action_mean = np.asarray(action_mean, np.float32)
        self.action_std = np.asarray(action_std, np.float32)

    @classmethod
    def fit(cls, states, actions):
        return cls(states.mean(0), states.std(0) + 1e-6, actions.mean(0), actions.std(0) + 1e-6)

    def norm_state(self, s):
        return (s - self.state_mean) / self.state_std

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a):
        return a * self.action_std + self.action_mean

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in
                ("state_mean", "state_std", "action_mean", "action_std")}

    @classmethod
    def from_dict(cls, d):
        return cls(d["state_mean"], d["state_std"], d["action_mean"], d["action_std"])


def episode_split(episode_ids, val_fraction=0.1, seed=0):
    """-> (train_episode_ids, val_episode_ids), deterministic."""
    eps = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(eps)
    n_val = max(1, int(round(len(eps) * val_fraction))) if len(eps) > 1 else 0
    return np.sort(eps[n_val:]), np.sort(eps[:n_val])


class ChunkDataset:
    """Windows of (image, state, language embedding, k actions, mask).

    Indexable and iterable in minibatches without torch's DataLoader so the
    whole thing stays in one process (the images are already in memory).
    """

    def __init__(self, flat, episodes, k, encoder, stats):
        self.k = int(k)
        self.stats = stats
        keep = np.isin(flat["episode"], episodes)
        self.idx = np.flatnonzero(keep)
        self.flat = flat
        self.n = len(self.idx)
        # chunk targets: for every frame, the next k actions within its episode
        ep = flat["episode"]
        last_of = {}
        for i in np.flatnonzero(keep):
            last_of[ep[i]] = i          # frames are stored in order, so the max wins
        self.last_of = last_of
        # embeddings once per unique sentence
        sentences = sorted(set(flat["tasks"][self.idx].tolist()))
        vecs = encoder.embed_all(sentences)
        self.embed_of = {s: vecs[i] for i, s in enumerate(sentences)}
        self.embed_dim = vecs.shape[1]

    def __len__(self):
        return self.n

    def window(self, j):
        i = int(self.idx[j])
        last = self.last_of[int(self.flat["episode"][i])]
        end = min(i + self.k, last + 1)
        acts = self.flat["actions"][i:end]
        mask = np.ones(self.k, np.float32)
        if acts.shape[0] < self.k:
            pad = np.repeat(acts[-1:], self.k - acts.shape[0], axis=0)
            mask[acts.shape[0]:] = 0.0
            acts = np.concatenate([acts, pad], axis=0)
        return {
            "image": self.flat["images"][i],
            "state": self.stats.norm_state(self.flat["states"][i]),
            "lang": self.embed_of[str(self.flat["tasks"][i])],
            "actions": self.stats.norm_action(acts),
            "mask": mask,
        }

    def batches(self, batch_size, rng=None, shuffle=True):
        order = np.arange(self.n)
        if shuffle:
            (rng or np.random.default_rng()).shuffle(order)
        for s in range(0, self.n, batch_size):
            ws = [self.window(j) for j in order[s:s + batch_size]]
            yield {key: np.stack([w[key] for w in ws]) for key in ws[0]}
