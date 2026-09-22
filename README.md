# so-arm100-vla

A small language-conditioned multitask policy for the
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) tabletop tasks,
trained from scratch on scripted demonstrations, evaluated on instructions
and (task, colour) pairs it never saw. CPU only. Built on
[`so-arm100-sim`](https://github.com/AungKaung1928/so-arm100-sim), which
defines the tasks, the expert, the success criteria and the instruction
templates this repository holds out.

**Status: code complete, tests green, no policy trained yet.** Every number
marked `TODO(measure)` is produced by the named command and filled in from
its JSON, never typed in.

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

### Results, 100 episodes x 5 seeds per (task, colour) cell, nominal physics

`python eval_language.py --ckpt runs/policy_minilm_s0.pt --episodes 100 --seeds 5`

| split | success | +- over seeds | reach | push | lift | pick_place |
|---|---|---|---|---|---|---|
| seen | TODO(measure) | | | | | |
| paraphrase | TODO(measure) | | | | | |
| combo | TODO(measure) | | | | | |

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
| MiniLM | TODO(measure) | | |
| hashed | TODO(measure) | | |
| MiniLM, no FiLM | TODO(measure) | | |

Three seeds each; the table reports mean and spread. `--no-film` removes the
language modulation of the vision features, leaving language as a token only.

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

| task | success (10 episodes) | forward pass p50 |
|---|---|---|
| reach / push / lift / pick_place | TODO(measure) | TODO(measure) s |

Caveats that make this a reference and not a result: different embodiment
normalisation statistics, different camera placement, radians against motor
units, 20 Hz control against 30 fps data. `--dry-run` validates the
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
