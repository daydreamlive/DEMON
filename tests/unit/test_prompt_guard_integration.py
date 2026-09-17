"""Ensure guards wrap both the greedy anchor and the final sampled candidate."""
from types import SimpleNamespace
import pytest
from demos.realtime_motion_graph_web import prompt_variations as pv

torch = pytest.importorskip("torch")


@pytest.fixture
def decoder(monkeypatch):
    class Encoding(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1
        decoded = "solo piano with drums"
        encoded = []
        def __call__(self, text, **kwargs):
            self.encoded.append(text)
            if kwargs.get("add_special_tokens") is False:
                return SimpleNamespace(input_ids=[2, 3])
            return Encoding(input_ids=torch.tensor([[2, 3, 1]]))
        def decode(self, ids, **kwargs):
            return self.decoded
        def convert_ids_to_tokens(self, ids):
            return ["▁word" for _ in ids]
        def __len__(self):
            return 8

    tok = Tokenizer()
    model = SimpleNamespace(generate=lambda **kwargs: torch.tensor([[0, 2, 1]]))
    monkeypatch.setattr(pv, "_load", lambda: (tok, model, "cpu"))
    return tok, model


def test_enhancement_rejects_extra_instrument(decoder):
    assert pv.enhance("solo piano") == "solo piano"


def test_enhancement_and_variations_reject_invented_rhythm(decoder, monkeypatch):
    source = "solo electric guitar, single instrument, stratvema, classic 70s rock"
    tok, _ = decoder
    tok.decoded = "unaccompanied solo electric guitar, driving syncopated rhythm"
    assert pv.enhance(source, deck="sa3") == source
    monkeypatch.setattr(pv, "_anchor", lambda *a: ([5, 6], source))
    monkeypatch.setattr(pv, "_sample", lambda *a, **kw: torch.ones((pv.LANES, 2), dtype=torch.long))
    assert pv.point(source, stop=8, lane=5, deck="sa3") == source


def test_invalid_greedy_anchor_cannot_become_forced_prefix(decoder, monkeypatch):
    tok, _ = decoder
    monkeypatch.setattr(pv, "_anchor", lambda *a: ([5, 6], "solo guitar"))
    monkeypatch.setattr(pv, "_sample", lambda *a, **kw: torch.ones((pv.LANES, 2), dtype=torch.long))
    assert pv.point("solo piano", stop=8, lane=5) == "solo piano"
    assert tok.encoded[-1] == "solo piano"


def test_home_preserves_exact_anchor_without_decode(decoder):
    _, model = decoder
    model.generate = lambda **kwargs: pytest.fail("Home must not re-enhance")
    assert pv.point("solo piano, intimate recording", stop=0) == "solo piano, intimate recording"


def test_conflicting_anchor_does_not_sample_an_invalid_fallback(decoder, monkeypatch):
    monkeypatch.setattr(pv, "_anchor", lambda *a: ([5, 6], "solo guitar"))
    monkeypatch.setattr(pv, "_sample", lambda *a, **kw: pytest.fail("Invalid anchor sampled"))
    assert pv.point("solo piano and drums", stop=8) == ""
