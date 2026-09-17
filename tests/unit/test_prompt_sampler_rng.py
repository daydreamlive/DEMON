"""Exercise real Transformers decoding without downloads or audio hardware."""
import threading
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from demos.realtime_motion_graph_web import prompt_variations as pv


def test_private_sampling_is_repeatable_under_concurrent_audio_rng():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = transformers.T5ForConditionalGeneration(transformers.T5Config(
            vocab_size=40, d_model=16, d_kv=4, d_ff=32, num_layers=1,
            num_decoder_layers=1, num_heads=2, decoder_start_token_id=0,
            eos_token_id=1, pad_token_id=0, dropout_rate=0.0,
        )).eval()
        enc = {"input_ids": torch.tensor([[3, 4, 1]]).repeat(3, 1)}
        def sample(seed):
            with torch.inference_mode():
                return pv._sample(torch, model, enc, [3, 4, 5, 6], 15, 16, seed, "cpu", 3)
        before = torch.get_rng_state().clone()
        baseline = {seed: sample(seed) for seed in (17, 91)}
        assert torch.equal(before, torch.get_rng_state())
        private = torch.Generator().manual_seed(731)
        expected = torch.randn(512, generator=private)
        done = threading.Event()
        errors = []
        def noise():
            while not done.wait(0.001):
                torch.manual_seed(731)
                if not torch.equal(expected, torch.randn(512)):
                    errors.append(True)
        worker = threading.Thread(target=noise, daemon=True)
        worker.start()
        try:
            for seed in (91, 17, 17, 91):
                assert torch.equal(sample(seed), baseline[seed])
        finally:
            done.set()
            worker.join(5)
        assert not worker.is_alive() and not errors
    finally:
        torch.set_num_threads(original_threads)


def test_fork_ties_are_stable_and_do_not_touch_global_rng():
    logits = torch.ones(30)
    before = torch.get_rng_state().clone()
    a = pv._fork_tokens(torch, logits, 12, 0.7)
    assert torch.equal(before, torch.get_rng_state())
    assert a == pv._fork_tokens(torch, logits, 12, 0.7)
    assert len(set(a)) == 12
