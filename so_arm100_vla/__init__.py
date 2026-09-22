"""so_arm100_vla -- a small language-conditioned multitask policy for the
SO-ARM100 tabletop tasks, trained from scratch on scripted demonstrations.

Not a vision-language-action foundation model and not a fine-tune of one.
The language side is a frozen sentence encoder (or a hashed bag of words, as
the ablation); the policy is a 128-wide transformer over three tokens that
emits chunks of joint targets.
"""
__version__ = "0.1.0"
