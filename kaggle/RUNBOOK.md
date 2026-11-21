# Kaggle run: SmolVLA fine-tune

One Kaggle session produces `smolvla_sim.zip`; everything after the download runs on the laptop.

## Before the click (laptop, under a minute, one core)

```
python kaggle/preflight.py      # Python >= 3.12 with lerobot 0.6.1 and so-arm100-sim
```

It records 1 episode for 2 pairs with the notebook's own record command, parses the notebook's
`lerobot-train` arguments with lerobot 0.6.1, and loads one batch (action chunk 50 x 6).
It does not load the model: GPU memory at batch 32 and s/step are first measured on Kaggle.
It uses the installed so-arm100-sim, while the notebook installs commit `4f9b6b1`; the recording path
(`record.py`, `smolvla_export.py`, `instructions.py`, `expert.py`, `env.py`, `render.py`) has no commit after it
as of 2026-11-22 (`git log 4f9b6b1..HEAD -- <those files>` is empty), and the same check passed once
on an archive of `4f9b6b1` itself.

## Clicks

1. kaggle.com, signed in. **Create -> New Notebook**, then **File -> Import Notebook** and upload
   `kaggle/smolvla_finetune.ipynb`.
2. Right panel, **Session options**: Accelerator **GPU T4 x2** (one GPU is used), Internet **on**.
   Both need the phone-verified account.
3. **Save Version -> Save & Run All (Commit)**. The run continues with the browser closed.
   Do not use the interactive "Run All": an idle interactive session is stopped and its files are lost.
4. When the version shows **Complete**: open it, **Output** tab, download `smolvla_sim.zip`,
   `train.log` and `timing.log`.

## Expected time (estimates until the first run; nothing here is measured on Kaggle)

| Step | Estimate | Basis |
|---|---|---|
| Install (venv, lerobot, torch) | 5-10 min | download size, unmeasured |
| Record 180 episodes (9 pairs x 20) | 45-90 min | laptop: 93 frames in 27 s on one core, CPU only |
| Timing run, 20 steps | 5-10 min | includes the model download; up to 3 tries (batch 32, 16, 8) |
| Fine-tune | 2.5 h | fixed by `TRAIN_HOURS`; the step count is sized from the timing run |
| Zip | 1-3 min | about 1.8 GB (450M float32 parameters), stored without compression |

Total about 4-5 h of the 12 h session limit and of the 30 h weekly GPU quota.

## If it fails

| Sign | Cause | Do |
|---|---|---|
| Cell 2 stops at `pip`/`uv` | Internet off | turn Internet on, run again |
| `nvidia-smi` prints nothing | no GPU selected | set the accelerator, run again |
| `timing run failed at every batch size` | not a memory problem if batch 8 fails too | read `timing.log` in the Output tab |
| `training failed` | see the last lines printed | `train.log` in the Output tab has the full log |

`train.log` is written while training runs, so it survives a stopped session.
The venv, dataset and checkpoints are in `/kaggle/temp` and are not saved: only the zip and the two logs are.

## Check the result (laptop)

```
mkdir -p runs/kaggle_t4 && unzip smolvla_sim.zip -d runs/kaggle_t4      # runs/* is git-ignored
grep -c "updt_s" runs/kaggle_t4/train/train.log                          # log lines, one per 50 steps
tail -3 runs/kaggle_t4/train/train.log                                   # last step = planned steps, loss
python eval_smolvla_ft.py --ckpt runs/kaggle_t4/train/checkpoints/last/pretrained_model \
    --episodes 1 --splits combo --out runs/smolvla_ft_smoke.json         # smallest first
```

Evaluation is CPU inference at about 3.16 s per 50-step chunk on 8 threads (measured, README), so the smoke is
1 episode on the held-out pairs; the full table (`--episodes 10`, all splits) takes hours
at that rate: run it thread-capped and in the background. Label training numbers "Kaggle T4" and evaluation numbers
with the laptop CPU and thread count.

A fine-tune counts as working when the loss in `train.log` falls and the seen-pair success is
above the zero-shot 0.00; the held-out `combo` split is the number that answers the project's
question (does the colour word bind to a task it was never paired with).
