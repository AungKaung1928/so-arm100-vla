"""Every README number that has a tracked runs/*.json is recomputed from that file.

The edge tables (OpenVINO, INT8, camera and step sweeps) have no tracked JSON and are not covered.
"""
import json
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# README.md is the short page; the full write-up it links lives in docs/RESULTS.md.
README = "".join(open(os.path.join(ROOT, f), encoding="utf-8").read() for f in ("README.md", "docs/RESULTS.md"))
SPLITS = ("seen", "paraphrase", "combo")
ROWS = {"minilm": "`LangPolicy`, MiniLM encoder, FiLM", "hash": "`LangPolicy`, hashed bag-of-words encoder",
        "minilm_nofilm": "`LangPolicy`, MiniLM, no FiLM"}


def run(name):
    with open(os.path.join(ROOT, "runs", name + ".json")) as f:
        return json.load(f)


def cell(r):
    return f"{r['success_rate']:.3f} ± {r['success_std_mean']:.3f}"


@pytest.mark.parametrize("tag", ROWS)
def test_at_a_glance_row(tag):
    cells = " | ".join(cell(run(f"eval_policy_{tag}_s0_{s}")) for s in SPLITS)
    assert f"| {ROWS[tag]} | {cells} |" in README


@pytest.mark.parametrize("split", SPLITS)
def test_minilm_results_row_and_per_task(split):
    r = run(f"eval_policy_minilm_s0_{split}")
    line = next(l for l in README.splitlines()
                if l.startswith(f"| `LangPolicy`, MiniLM | success, `{split}` |"))
    assert f"| {cell(r)} |" in line
    assert (r["n_episodes"], len(r["seeds"])) == (20, 5)      # "20 episodes × 5 evaluation seeds"
    if split != "combo":
        want = ", ".join(f"{t} {r['by_task'][t]:.2f}" for t in ("reach", "push", "lift", "pick_place"))
        assert want in line


def test_smolvla_zero_shot_line():
    r = run("smolvla_zero_shot")
    wins = sum(round(c["success_rate"] * c["n_episodes"]) for c in r["cells"].values())
    total = sum(c["n_episodes"] for c in r["cells"].values())
    assert f"{wins} of {total} episodes" in README
    assert (f"| {r['forward_s_p50']:.2f} s (p90 {r['forward_s_p90']:.2f} s, n = {r['n_forward']}) |"
            in README)


def test_lora_gate_line():
    r = run("lora_gate")
    line = next(l for l in README.splitlines() if l.startswith("| LoRA gate"))
    share = 100 * r["params_trainable"] / r["params_total"]
    for want in (f"**{r['step_s_best']:.2f} s, best of {len(r['step_s'])} → {r['verdict']}**",
                 f"GO under {r['gate_s']:.0f} s/step", f"{r['params_total'] / 1e6:.1f}M stand-in",
                 f"{r['params_trainable'] / 1e6:.2f}M trainable ({share:.2f}%)",
                 f"batch 1, seq {r['seq']}, {r['threads']} threads"):
        assert want in line, want


def test_edge_notes_quote_only_timings_the_readme_states():
    notes = open(os.path.join(ROOT, "edge", "NOTES.md"), encoding="utf-8").read()
    quoted = set(re.findall(r"(?<![\d,])(?:\d,)?\d{3} ms", notes))
    assert "733 ms" in quoted and "3,770 ms" in quoted
    missing = {q for q in quoted if not re.search(rf"(?<![\d,]){q[:-3]}(?!\d)", README)}
    assert not missing, missing


def test_floor_line():
    r = run("eval_floor_hold")["splits"]
    episodes = sum(len(s["cells"]) * s["n_episodes"] * len(s["seeds"]) for s in r.values()
                   if s["split"] != "paraphrase")        # same pairs as seen: counted once
    assert all(s["success_rate"] == 0.0 for s in r.values())
    assert f"0 of {episodes:,} episodes; `runs/eval_floor_hold.json`" in README


@pytest.mark.parametrize("tag,name", [("minilm", "MiniLM, FiLM"), ("hash", "hashed bag-of-words"),
                                      ("minilm_nofilm", "MiniLM, no FiLM")])
def test_language_probe_row(tag, name):
    p = run(f"probe_language_{tag}_s0")
    keys = ("l1_recorded", "l1_wrong_colour", "l1_wrong_task", "l1_other_image")
    cells = " | ".join(f"{p[s][k]:.4f}" for s in ("val", "train") for k in keys)
    assert f"| {name} | {cells} |" in README
    # recorded sentences reproduce the training log's final val L1
    assert abs(p["val"]["l1_recorded"] - run(f"policy_{tag}_s0")["trace"][-1]["val_l1"]) < 1e-4
    # the README's interval claim: held-out colour/image intervals contain 0, task excludes it
    v = p["val"]
    assert v["ci95_change_wrong_colour"][0] < 0 < v["ci95_change_wrong_colour"][1]
    assert v["ci95_change_other_image"][0] < 0 < v["ci95_change_other_image"][1]
    assert v["ci95_change_wrong_task"][0] > 0
    assert "includes zero for the wrong\ncolour and the other image in all three policies, and excludes it for the wrong task" in README


def test_cpp_step_table():
    r = run("act_cpp")
    rounds = r["step_ms_rounds"]

    def span(key, fmt="{:.2f}"):
        v = [x[key] for x in rounds]
        return f"{fmt.format(min(v))}-{fmt.format(max(v))}"

    assert r["rounds"] == len(rounds) == 5 and r["n_time"] == 1000 and r["threads"] == 1
    assert f"| PyTorch 2.11 eager | {span('torch_p50')} ms |" in README
    assert f"| Python + ONNX Runtime | {span('python_ort_p50')} ms |" in README
    assert f"| C++ + ONNX Runtime | {span('cpp_p50')} ms |" in README
    assert f"over {r['n_parity']} chunks: {r["max_abs_cpp_vs_torch"]:.1e}" in README
    assert r["max_abs_cpp_vs_torch"] <= r["tol"]
    ratio = [x["torch_p50"] / x["python_ort_p50"] for x in rounds]
    assert f"ONNX Runtime is {min(ratio):.1f}-{max(ratio):.1f}x per round" in README
    gap = [x["python_ort_p50"] - x["cpp_p50"] for x in rounds]
    assert f"saves\n{min(gap):.2f}-{max(gap):.2f} ms" in README
    assert f"between\n{span('cpp_p99')} ms in C++" in README


def test_cpp_int8_table():
    r = run("act_cpp_int8")
    for x in r["rows"]:
        name = f"{x['calib_frames']}, {'all INT8' if x['variant'] == 'all INT8' else 'output layer fp32'}"
        assert (f"| {name} | {x['max_abs_rad']:.3f} / {x['mean_abs_rad']:.3f} rad | "
                f"{x['cpp_step_ms_p50']:.2f} ms |") in README
    assert f"fp32 took {r['fp32_cpp_step_ms_p50']:.2f} ms in the same run" in README
    lo, hi = r["calib_norm_action_range"]["32"]
    assert f"(32 frames: {lo:.2f} to {hi:.2f} normalised; the evaluated chunks reach\n{r['eval_norm_action_range'][1]:.2f})" in README
    assert hi < r["eval_norm_action_range"][1]          # the clipping the text explains
    best = min(x["cpp_step_ms_p50"] for x in r["rows"])
    assert f"saves at most {r['fp32_cpp_step_ms_p50'] - best:.2f} ms" in README
    assert f"by up to {max(x['max_abs_rad'] for x in r['rows']):.2f} rad" in README


def test_closed_loop_paragraph():
    runs = {"reach:red": "act_closed_loop", "reach:blue": "act_closed_loop_reach_blue"}
    eps, maxd = [], 0.0
    for task, name in runs.items():
        r, c = run(name), run(name + "_control")
        assert r["task"] == c["task"] == task and r["episodes"] == c["episodes"] == 8
        assert c["max_action_diff"] == 0.0                     # "differs by exactly 0 on both"
        eps += r["per_episode"]
        maxd = max(maxd, r["max_action_diff"])
    red, blue = run("act_closed_loop"), run("act_closed_loop_reach_blue")
    assert f"same\noutcome in {red['same_outcome']} of 8, {red['success_torch']} successes each" in README
    assert red["success_torch"] == red["success_onnx"]
    assert (f"{blue['same_outcome']} of 8, {blue['success_torch']} successes against "
            f"{blue['success_onnx']}") in README
    first = [e["first_step_over_1e-3"] for e in eps if e["first_step_over_1e-3"] is not None]
    assert f"in {len(first)} of {len(eps)} episodes" in README
    assert f"after {min(first)}-{max(first)} steps, by up to {maxd:.2f} rad" in README


def test_demo_caption_is_the_closed_loop_json():
    cl = json.load(open(os.path.join(ROOT, "runs", "act_closed_loop.json")))
    assert cl["ckpt"] == "runs/policy_hash_s0.pt" and cl["task"] == "reach:red"
    assert f"It solved {cl['success_torch']} of {cl['episodes']} such episodes" in README
    assert os.path.getsize(os.path.join(ROOT, "docs", "media", "demo.gif")) < 3_000_000


def test_smolvla_and_l1_rows():
    z = run("smolvla_zero_shot")
    succ = [c["success_rate"] for c in z["cells"].values()]
    n = sum(c["n_episodes"] * len(c["seeds"]) for c in z["cells"].values())
    ratio = z["forward_s_p50"] / 0.05                        # 20 Hz control period
    assert f"{z['forward_s_p50']:.2f} s (p90 {z['forward_s_p90']:.2f} s, n = {z['n_forward']})" in README
    assert f"0 of {n}" in README and not any(succ)
    assert f"{ratio:.0f}x the 50 ms control period" in README and "64x" not in README
    last = run("policy_minilm_s0")["trace"][-1]
    assert f"| {last['train_l1']:.3f} / {last['val_l1']:.3f} |" in README


def test_minilm_cosines():
    c = run("minilm_cosines")
    para = [v for _, v in c["paraphrases"]]
    assert f"| {min(para):.2f}–{max(para):.2f} / **{c['other_task'][1]:.2f}** / {c['other_colour'][1]:.2f} |" in README
    assert f"is {c['extra_paraphrase'][1]:.2f} from its paraphrase\n\"{c['extra_paraphrase'][0].rsplit(' ', 2)[0]}" in README
    hoist = dict(c["paraphrases"])["hoist the green block"]
    take = dict(c["paraphrases"])["take the green cube and hold it up"]
    assert f'only {hoist:.2f} from its own paraphrase "hoist the green block" and {take:.2f} from' in README


def test_policy_size():
    n = run("policy_minilm_s0")["params"]
    assert f"{n / 1e6:.1f}M param" in README and f"{n:,}" in README
    assert "about 1M" not in README and " 1M-param" not in README


def test_eval_wall_time_names_its_process_count():
    """1.7 h was measured with --jobs 8 (docs/RESULTS.md command); a bare 1.7 h misleads a one-process run."""
    for line in (l for l in README.splitlines() if re.search(r"1\.7 h(?!our)", l)):
        assert "jobs 8" in line, line
