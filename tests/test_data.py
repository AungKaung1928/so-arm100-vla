import numpy as np

from so_arm100_vla import text as T
from so_arm100_vla.data import ChunkDataset, Stats, episode_split


def synthetic_flat(n_eps=3, lengths=(7, 12, 5), h=8, w=10):
    states, actions, episode, frame, tasks, images = [], [], [], [], [], []
    for e, L in enumerate(lengths[:n_eps]):
        for t in range(L):
            states.append(np.full(15, e + 0.1 * t, np.float32))
            actions.append(np.full(6, 10 * e + t, np.float32))
            episode.append(e)
            frame.append(t)
            tasks.append(["pick up the red cube", "push the blue cube to the target"][e % 2])
            images.append(np.full((h, w, 3), e, np.uint8))
    return {"states": np.stack(states), "actions": np.stack(actions),
            "episode": np.array(episode), "frame": np.array(frame),
            "tasks": np.array(tasks), "images": np.stack(images)}


def test_episode_split_is_deterministic_and_disjoint():
    ids = np.repeat(np.arange(20), 5)
    tr, va = episode_split(ids, 0.1, seed=3)
    tr2, va2 = episode_split(ids, 0.1, seed=3)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)
    assert len(va) == 2 and not set(tr) & set(va) and len(tr) + len(va) == 20


def test_windows_pad_with_last_action_and_mask():
    flat = synthetic_flat()
    stats = Stats(np.zeros(15), np.ones(15), np.zeros(6), np.ones(6))
    ds = ChunkDataset(flat, episodes=np.array([0, 1]), k=4, encoder=T.HashEncoder(), stats=stats)
    assert len(ds) == 7 + 12
    # episode 0 has 7 frames: window at its frame 5 sees actions 5,6 then pads 6,6
    j = int(np.flatnonzero((flat["episode"][ds.idx] == 0) & (flat["frame"][ds.idx] == 5))[0])
    wdw = ds.window(j)
    assert wdw["actions"][:, 0].tolist() == [5, 6, 6, 6]
    assert wdw["mask"].tolist() == [1, 1, 0, 0]
    # a window never crosses into the next episode
    j = int(np.flatnonzero((flat["episode"][ds.idx] == 0) & (flat["frame"][ds.idx] == 6))[0])
    assert ds.window(j)["actions"][:, 0].tolist() == [6, 6, 6, 6]
    assert ds.window(0)["image"].shape == (8, 10, 3)


def test_embeddings_computed_once_per_sentence():
    flat = synthetic_flat()

    class Counting(T.HashEncoder):
        calls = 0

        def embed_all(self, s):
            Counting.calls += 1
            return super().embed_all(s)

    stats = Stats.fit(flat["states"], flat["actions"])
    ds = ChunkDataset(flat, episodes=np.array([0, 1, 2]), k=3, encoder=Counting(), stats=stats)
    assert Counting.calls == 1 and len(ds.embed_of) == 2
    b = next(ds.batches(8, rng=np.random.default_rng(0)))
    assert b["lang"].shape == (8, 384) and b["actions"].shape == (8, 3, 6)
    # normalised training actions have ~zero mean over the whole set
    allw = np.concatenate([w["actions"][:1] for w in (ds.window(j) for j in range(len(ds)))])
    assert abs(allw.mean()) < 0.3


def test_stats_round_trip():
    s = Stats(np.ones(15), np.full(15, 2.0), np.ones(6), np.full(6, 4.0))
    a = np.arange(6, dtype=np.float32)
    assert np.allclose(s.denorm_action(s.norm_action(a)), a)
    s2 = Stats.from_dict(s.to_dict())
    assert np.allclose(s2.action_std, 4.0)
