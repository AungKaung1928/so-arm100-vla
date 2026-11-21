# What limits each backend

SmolVLA-base (450M), laptop CPU, 4 threads, one 50-step action chunk = 2.5 s of motion at 20 Hz.
Every number is from the README edge section; none has a tracked JSON yet (see "Open" below).

| Backend | Measured (p50 per chunk) | What limits it | What is not known |
|---|---|---|---|
| PyTorch FP32 | 3 cameras 3,770 ms interleaved; 1 camera 1,848 ms | The vision encoder: 1,521 ms of a 3,154 ms chunk (48 %) for 3 cameras at 512 px. Denoising steps are 13 %. | Nothing about accuracy: it is the reference. |
| OpenVINO FP32 | 3 cameras 3,008 ms; 1 camera 1,398 ms | Same graph, same precision: 20-24 % faster than PyTorch, outputs equal to 3.9e-6. Still over the 2.5 s chunk at 3 cameras. Input shapes are fixed at export, so every camera count or resolution is a new export. | Whether a fine-tuned checkpoint exports the same way. |
| OpenVINO INT8 weights | 3 cameras 1,295 ms; 1 camera 733 ms | Accuracy, not time: output differs from PyTorch by 0.07-0.16 per chunk, the same size as the change between two different inputs (0.18). | Closed-loop success. Open-loop error on a zero-shot model cannot decide it; needs the fine-tuned checkpoint. |
| ONNX Runtime | not run | The export library (physicalai-train 0.3.0, `SmolVLA.get_supported_export_backends()`) lists only `torch` and `openvino`. (The small ACT policy in so-arm100-il does have an ONNX Runtime row.) | Everything. |

## Levers, in order of measured effect

1. Input size: 3 cameras 512 px 3,020 ms, 1 camera 1,360 ms, 3 cameras 256 px 1,107 ms. The model was not trained at those inputs: needs a fine-tune at the reduced input.
2. INT8 weights: about 2x at either input size.
3. Backend (PyTorch to OpenVINO FP32): 20-24 %.
4. Denoising steps 10 to 1: about 0.6 s interleaved, at a change of 0.5 to 0.7 in the chunk. Smallest gain, largest change.

## Method rule learned here

Rows timed in separate processes hid a 20 % backend gain and produced one wrong claim (that the
step count does not matter). Compare backends call by call in one process (`edge/*_interleaved.py`,
`edge/combo_1cam.py`).

## Open

- No edge row is backed by a tracked `runs/*.json`, so `tests/test_readme.py` cannot check them.
  `edge/combo_1cam.py <ov_dir> 20 runs/combo_1cam.json` writes one (a multi-minute, multi-thread run);
  the README row must then be updated to that run, since timings move between sessions.
- INT8 closed-loop success on the fine-tuned checkpoint.
- This is a laptop CPU, not an embedded board: none of these numbers transfer to a Jetson.
