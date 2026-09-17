"""Checkpoint initialization must not corrupt concurrent audio parameters."""

from threading import Event, Thread

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from demos.realtime_motion_graph_web import prompt_variations as pv


def test_cold_t5_load_preserves_concurrent_parameters(tmp_path, monkeypatch):
    # A tiny, real checkpoint exercises from_pretrained's initialization and
    # materialization without a network download or a GPU.
    config = transformers.T5Config(
        vocab_size=16, d_model=16, d_kv=4, d_ff=32,
        num_layers=1, num_decoder_layers=1, num_heads=2,
        decoder_start_token_id=0, eos_token_id=1, dropout_rate=0.0,
    )
    baseline = transformers.T5ForConditionalGeneration(config).eval()
    baseline.save_pretrained(tmp_path)
    monkeypatch.setenv("DEMON_ENHANCER_DIR", str(tmp_path))
    monkeypatch.setenv("DEMON_ENHANCER_DEVICE", "cpu")
    monkeypatch.setattr(pv, "_loaded", None)
    monkeypatch.setattr(pv, "_load_failed", False)
    monkeypatch.setattr(pv, "_retry_after", 0.0)
    tokenizer = object()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained",
        lambda *a, **kw: tokenizer,
    )

    entered, release = Event(), Event()
    original_init = transformers.T5ForConditionalGeneration.__init__

    def hold_constructor(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        entered.set()
        assert release.wait(15), "Constructor was not released"

    monkeypatch.setattr(
        transformers.T5ForConditionalGeneration, "__init__", hold_constructor,
    )
    register = torch.nn.Module.register_parameter
    initializer = torch.nn.init.uniform_
    layer = torch.nn.Linear(8, 4, bias=False)
    weight = layer.weight.detach().clone()
    rng = torch.get_rng_state().clone()
    result = []
    worker = Thread(target=lambda: result.append(pv._load()), daemon=True)
    worker.start()
    try:
        assert entered.wait(15), "T5 constructor was not reached"
        assert torch.nn.Module.register_parameter is register
        assert torch.nn.init.uniform_ is initializer
        assert torch.equal(torch.get_rng_state(), rng)
        # LoRA registration re-registers an existing audio weight as
        # parametrizations.weight.original, exposing the global meta race.
        torch.nn.utils.parametrize.register_parametrization(
            layer, "weight", torch.nn.Identity(), unsafe=True,
        )
        assert not layer.parametrizations.weight.original.is_meta
        assert torch.equal(layer.weight, weight)
        torch.nn.utils.parametrize.remove_parametrizations(
            layer, "weight", leave_parametrized=False,
        )
        assert torch.equal(layer.weight, weight)
    finally:
        release.set()
        worker.join(15)

    assert not worker.is_alive()
    assert result and result[0] is not None
    tok, loaded, device = result[0]
    assert tok is tokenizer and device == "cpu"
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name
    assert loaded.shared.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    inputs = torch.tensor([[2, 3, 1]])
    with torch.inference_mode():
        assert torch.equal(
            baseline.generate(inputs, do_sample=False, max_new_tokens=5),
            loaded.generate(inputs, do_sample=False, max_new_tokens=5),
        )
