import numpy as np
import pytest

from so_arm100_vla import text as T


def test_hash_encoder_is_deterministic_and_384d():
    e1, e2 = T.HashEncoder(), T.HashEncoder()
    a = e1.embed_all(["pick up the red cube", "push the blue block to the target"])
    b = e2.embed_all(["pick up the red cube", "push the blue block to the target"])
    assert a.shape == (2, 384) and a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def test_hash_encoder_separates_colour_words():
    e = T.HashEncoder()
    r, b = e.embed_all(["pick up the red cube", "pick up the blue cube"])
    same = e.embed(" Pick up the RED cube ")
    assert not np.allclose(r, b)
    assert np.allclose(r, same)          # case and whitespace do not matter


def test_make_encoder_names():
    assert T.make_encoder("hash").name == "hash"
    with pytest.raises(KeyError):
        T.make_encoder("clip")


def _minilm_available():
    try:
        import sentence_transformers  # noqa: F401
        import socket
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except Exception:       # noqa: BLE001
        return False


@pytest.mark.skipif(not _minilm_available(), reason="sentence-transformers or network unavailable")
def test_minilm_encoder_and_cache(tmp_path):
    enc = T.MiniLMEncoder(cache_dir=str(tmp_path))
    v = enc.embed_all(["lift the green cube", "raise the green block into the air"])
    assert v.shape == (2, 384)
    # Unit norm, and a paraphrase is clearly related. NOT asserted: that a
    # paraphrase is closer than a different-task sentence with shared words --
    # MiniLM rates "push the red cube to the target" about as close to "lift
    # the green cube" as its own paraphrase, which is the encoder's real
    # behaviour on these templates and one reason the paraphrase split is hard.
    assert abs(float(v[0] @ v[0]) - 1.0) < 1e-4
    assert float(v[0] @ v[1]) > 0.35
    # second encoder instance reads the cache, no model load
    enc2 = T.MiniLMEncoder(cache_dir=str(tmp_path))
    v2 = enc2.embed_all(["lift the green cube"])
    assert enc2._model is None and np.allclose(v2[0], v[0])
