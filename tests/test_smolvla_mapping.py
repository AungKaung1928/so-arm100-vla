import numpy as np

from smolvla_reference import StubPolicy, SmolVLARunner, build_batch, resize_pad


def test_resize_pad_letterboxes():
    img = np.full((96, 128, 3), 200, np.uint8)
    x = resize_pad(img, 64)
    assert x.shape == (1, 3, 64, 64)
    # 96x128 -> 48x64 inside 64x64: rows 0-7 and 56-63 are padding
    assert float(x[0, :, :8].abs().max()) == 0.0 and float(x[0, :, 8:56].mean()) > 0.7


def test_build_batch_matches_stub_features():
    pol = StubPolicy()
    obs = {"state": np.arange(15, dtype=np.float32), "image": np.zeros((96, 128, 3), np.uint8)}
    batch = build_batch(obs, "pick up the red cube", pol.config.input_features,
                        pol.config.resize_imgs_with_padding)
    assert set(batch) == {"task", "observation.images.camera1", "observation.images.camera2",
                          "observation.state"}
    assert batch["observation.state"].tolist() == [[0, 1, 2, 3, 4, 5]]
    a = pol.select_action(batch)
    assert tuple(a.shape) == (1, 6)


def test_runner_drives_the_stub_through_the_protocol():
    from so_arm100_sim.evaluate import evaluate
    from so_arm100_sim import render
    import pytest
    if not render.available():
        pytest.skip("no GL")
    pol = StubPolicy()
    times = []
    r = evaluate(lambda env: SmolVLARunner(pol, ["pick up the red cube"], log_time=times),
                 "lift:red:distractors", n_episodes=1, seeds=(0,),
                 env_kwargs={"obs_mode": "proprio", "action_mode": "absolute",
                             "image": {"camera": "front", "height": 24, "width": 32}})
    assert r["success_rate"] == 0.0 and len(times) == r["steps_mean"]
