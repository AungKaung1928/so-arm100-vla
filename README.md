# so-arm100-vla

A small language-conditioned multitask policy for the
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) tabletop tasks,
trained from scratch on scripted demonstrations, evaluated on instructions
and (task, colour) pairs it never saw. CPU only. Built on
[`so-arm100-sim`](https://github.com/AungKaung1928/so-arm100-sim), which
defines the tasks, the expert, the success criteria and the instruction
templates this repository holds out.

**Status: measured 2026-09-24.** The trained policy is weak (11% on seen
instructions, 3% on held-out combinations) and the encoder ablation is
inside its own noise. Every number below is produced by the named command
and copied from its JSON in `runs/`, never typed in.

**Walkthrough:** https://aungkaung1928.github.io/projects/so-arm100.html — the bench and the three policy projects built on it, explained end to end.

## What this is, and what it is not

It **is**: a 128-wide transformer over three tokens (a frozen sentence
embedding, 32 image keypoints from a small CNN, the arm's proprioception)
that emits chunks of ten joint targets, trained with an L1 loss on a few
thousand frames of expert demonstrations for four tasks and three colours.
The instruction decides which cube and which task; all three cubes are
always on the table.

It is **not** a vision-language-action foundation model, and it does **not**
fine-tune one. SmolVLA-base appears below as a zero-shot reference run
through the same protocol, inference only. A LoRA fine-tune of a
SmolVLM-sized network is gated on a measured compute check and is not
attempted unless the gate passes -- and even then it is future work, not a
result here.

Why build the small version at all: the interesting property of a VLA is
that one policy follows language across tasks and objects and generalises
to phrasings and combinations it was not trained on. That property can be
tested at 1M parameters on a laptop CPU with a real held-out protocol. What
cannot be tested here is scale, and this repository does not pretend to.

## The three splits

Sentences and (task, colour) combinations come from the bench's
`instructions` module, so "unseen" means the same thing here, in the
imitation project, and in the recorder that never writes the held-out side.

| split | templates | (task, colour) pairs | what success on it shows |
|---|---|---|---|
| `seen` | training | training (9 of 12) | the policy executes what it was shown |
| `paraphrase` | **held out** | training | the instruction is read through the encoder, not matched as a string |
| `combo` | training | **held out** (3 of 12) | colour and task compose; every task and every colour was seen, never together |

Held-out pairs: `lift:blue`, `push:red`, `pick_place:green`. Five training
and three held-out templates per task, e.g. training "pick up the {c} cube",
held-out "hoist the {c} block".

### Results, 20 episodes x 5 seeds per (task, colour) cell, nominal physics

`python eval_language.py --ckpt runs/policy_minilm_s0.pt --episodes 20 --seeds 5 --jobs 8`

The protocol was written as 100 episodes per seed. At about 1.7 hours per
policy for 20 on this CPU, 100 would have been over eight hours per policy
and 25 for the ablation, so every policy here ran 20 x 5; the spread over
seeds below is what that costs.

| split | success | +- over seeds | reach | push | lift | pick_place |
|---|---|---|---|---|---|---|
| seen | 0.110 | 0.081 | 0.25 | 0.04 | 0.07 | 0.01 |
| paraphrase | 0.097 | 0.060 | 0.24 | 0.03 | 0.03 | 0.03 |
| combo | 0.027 | 0.046 | — | 0.03 | 0.03 | 0.02 |

**What this says: the policy barely works.** Reach succeeds about a
quarter of the time; push, lift and pick-place sit between 0 and 7%, which
is close to the floor. This is the measured result, not a placeholder. The
training record says why: the final-epoch L1 on held-out demonstration
episodes is 0.128 against 0.057 on the training ones (`runs/policy_minilm_s0.json`),
from 162 training episodes, about 20 per (task, colour) pair. The policy
fits the demonstrations it saw and does not generalise from them, before
language even enters. The next experiment is more demonstrations per pair,
not a different encoder.

Reference line: the scripted expert that produced the data succeeds at
roughly 0.9 or better on `lift` and `pick_place`, 0.8-0.9 on `push`, 1.0 on
`reach` in the bench's development readings; its measured table lives in
the bench repository. A learned policy is bounded by it.

## The encoder ablation: does the language model matter?

Two encoders produce the same 384-d embedding shape:

- **MiniLM** (`all-MiniLM-L6-v2`, 22M parameters, frozen). A sentence model
  that puts "hoist the blue block" near "lift the blue cube".
- **Hashed bag of words + bigrams.** No parameters, no semantics. "hoist"
  and "lift" land in unrelated buckets.

If the hashed policy scores as well as MiniLM on `paraphrase`, the split was
not testing language understanding. If MiniLM wins there and ties on
`seen`, the difference is the encoder's contribution and nothing else,
because the policy, data and seeds are identical.

| encoder | seen | paraphrase | combo |
|---|---|---|---|
| MiniLM | 0.110 ± 0.081 | 0.097 ± 0.060 | 0.027 ± 0.046 |
| hashed | 0.092 ± 0.046 | 0.073 ± 0.057 | 0.027 ± 0.021 |
| MiniLM, no FiLM | 0.137 ± 0.075 | 0.110 ± 0.056 | 0.033 ± 0.038 |

Success rate, mean ± standard deviation over the 5 evaluation seeds, one
training seed per encoder (the protocol said three; one training run is
about four minutes, but each evaluation is 1.7 hours, and three would have
tripled that). `--no-film` removes the language modulation of the vision
features, leaving language as a token only.

**The ablation does not separate the encoders.** Every row is inside the
others' spread, and the ordering (no FiLM highest on `seen`) is noise at
this success level: with the base policy at 3-14%, there is no headroom for
the encoder to show a difference. The test is valid; the policy under it is
too weak for it to have power. It becomes informative once the base policy
is fixed by more data, and not before.

One measured fact about MiniLM on these templates, before any policy is
trained (cosine similarities, `test_text.py` records the behaviour rather
than asserting it away): "lift the green cube" is 0.59 from its paraphrase
"raise the green block into the air" and 0.38 from "hoist the green block",
but **0.63 from "push the red cube to the target"** and 0.82 from "lift the
blue cube". The encoder is driven by shared surface words; the colour word
moves the embedding very little. So the `paraphrase` split is hard for a
reason that sits in the encoder, and the `combo` split asks the policy to
read colour out of a small difference between two nearly parallel vectors.
Whether FiLM on the vision features is enough to do that is exactly what the
ablation measures.

## SmolVLA-base, zero-shot, same simulator

`python smolvla_reference.py --run --episodes 10`

`lerobot/smolvla_base` was pretrained on community SO-100/101 data: real arms,
real cameras, motor units. Here it receives a rendered 96x128 frame
letterboxed to its 512x512 input, six joint angles in radians zero-padded to
its state width, and the instruction. Nothing is adapted, and the expected
result is low. This is a **calibration line**, not a comparison of
capability: it says what a well-known model does under this protocol with
no adaptation, next to a small policy that was trained for it.

| task | success (10 episodes each) | chunk inference p50, CPU |
|---|---|---|
| reach / push / lift / pick_place | 0.00 / 0.00 / 0.00 / 0.00 | 3.16 s (p90 3.30 s, n = 130) |

0 of 40 episodes, as expected. The line is still useful for two things: it
shows the observation mapping runs end to end against the real checkpoint,
and it prices the model on this machine. One 50-step action chunk costs
3.2 s on 8 CPU threads, which is 64x the 50 ms control period at 20 Hz;
even with chunking the arm would wait 3 s every 2.5 s of motion. On CPU a
500M-parameter VLA is not a real-time controller, whatever its success
rate after fine-tuning.

Caveats that make this a reference and not a result: different embodiment
normalisation statistics, different camera placement, radians against motor
units, 20 Hz control against 30 fps data, and one rendered view copied
into all three of its camera slots. `--dry-run` validates the
observation mapping against a stub with the same feature keys and runs in
the tests without the download.

## The LoRA gate

`OMP_NUM_THREADS=8 nice -n 10 python lora_gate.py`

Before proposing a fine-tune, measure whether one step of it is affordable.
The gate builds a decoder with SmolVLM-256M's language-model shape (hidden
576, 30 layers, 9 heads, MLP 1536), attaches rank-8 LoRA to the query and
value projections (a from-scratch `LoRALinear`, unit tested: identity at
init, exact merge/unmerge), and times one forward-backward-optimizer step at
batch 1, sequence 256, on 8 threads. **Rule, fixed before measuring: GO
under 5 s/step, else NO-GO.** At 5 s/step, 2,000 steps is 2.8 hours, which is
the most a chunked, attended CPU run here can carry.

Measured 2026-09-22 (`runs/lora_gate.json`): stand-in of 176.7M parameters,
0.55M trainable (0.31%), **0.51 s per step on 8 threads, best of 2 -> GO**.
2,000 steps would be about 0.3 h. The gate measures compute feasibility of
the language-model half only: the real SmolVLA action expert adds a vision
tower and a flow-matching head that were not timed, and a GO here says a
fine-tune is affordable to *try*, not that it would help. A NO-GO would have
dropped the fine-tune from the project; it did not lower the bar.

## What is in the box

```
so_arm100_vla/
  text.py        MiniLMEncoder (cached to disk) and HashEncoder, both 384-d
  splits.py      seen / paraphrase / combo, from the bench's instruction tables
  data.py        recording via the bench expert; flat cache; k-step action windows
  model.py       LangPolicy (CNN -> FiLM -> keypoints; 3 tokens; k queries), ChunkEnsembler
  policy.py      LangRunner: carries the sentence the bench env does not know about
  lora.py        LoRALinear, attach_lora
record_data.py   every training combination, distractors on, successes only
train.py         final-epoch reporting, checkpoint carries stats + encoder name
eval_language.py the three splits through the bench protocol, JSON + table
smolvla_reference.py, lora_gate.py
tests/           encoders, splits, windowing, shapes, ensembling, LoRA math,
                 SmolVLA mapping, and an end-to-end record -> train -> eval smoke
```

Model size at the default configuration: printed by `train.py`
(`LangPolicy N params`), about 1M.

## Reproducing

```
python3 -m venv .venv && . .venv/bin/activate          # Python >= 3.12
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
./verify.sh                                            # tests, then the commands below
python record_data.py --root data/multitask --episodes 20
nice -n 10 python train.py --root data/multitask --encoder minilm --epochs 30
python eval_language.py --ckpt runs/policy_minilm_s0.pt
```

Recording renders every frame in software (about 20 ms at 96x128), so 20
episodes per combination is roughly ten minutes. Training is CPU-only,
under `nice`, chunked by epochs; the log warns when throughput drops more
than 20% from the first epochs, which on a laptop is either throttling or
another process, and the load average says which.

```
docker build -t so-arm100-vla . && docker run --rm so-arm100-vla   # offline tests
```

Image built on 2026-09-23 and its default command passed inside it (21 tests passed, 2 skipped, the one that downloads a text encoder deselected), image size 3.13 GB.

## Limits, stated

- Simulation only, fixed front camera, three cubes of three colours.
- Language is templated. Negation, relations and counting are outside the
  set (see `docs/ISSUES.md`).
- The expert is scripted; the policy inherits its style and its failure
  modes.
- The SmolVLA line is zero-shot from a different embodiment and is not
  evidence about SmolVLA.
- Nothing here has run on a physical arm.

## Licence

MIT. The bench and its vendored Menagerie model carry their own licences.
