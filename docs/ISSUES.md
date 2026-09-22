# Issues to open on GitHub

One block per issue: title, then body. The known gaps at the first release.

---

**LoRA fine-tune of SmolVLM is gated, not attempted**

`lora_gate.py` measures one training step of a SmolVLM-256M-shaped decoder
with rank-8 LoRA on the CPU. The decision rule is fixed at 5 s/step. If the
gate says GO, the next step is a real LoRA on the SmolVLA action expert with
the recorded dataset; if NO-GO, this issue closes as won't-fix and the README
keeps saying so. Either way no fine-tuning result exists today.

---

**Wrist camera**

The policy sees the fixed front camera only. A wrist camera would make the
colour word easier to ground at grasp time. Needs one `add_camera` in the
bench scene and a second image token here.

---

**Negation and relational instructions are not covered**

"pick up the cube that is not red" and "the cube next to the marker" are
outside the template set. The three splits test paraphrase and composition
of (task, colour) only. Adding relational templates changes the dataset and
the expert (which cube is "next to" needs a rule).

---

**MiniLM versus a CLIP text encoder**

MiniLM is a sentence-similarity model, not a vision-aligned one. A CLIP text
tower (ViT-B/32, 63M) is the natural second point on the encoder ablation.
Cost: a second download and a 512-d projection; the policy does not change.

---

**Object-centric evaluation with more distractors**

Three cubes, three colours. Success on the combo split with two distractors
does not say what happens with five objects or two cubes of the same colour.

---

**Per-task action-chunk length**

`k=10` is shared. `reach` finishes in ~12 steps, `pick_place` takes ~110; a
per-task or adaptive chunk length is untested.
