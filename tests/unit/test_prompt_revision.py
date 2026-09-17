import json
from types import SimpleNamespace
import pytest
from demos.realtime_motion_graph_web import server, prompt_enhancer as pe, prompt_variations as pv
from demos.realtime_motion_graph_web.prompt_identity import checkpoint_digest, revision_for


def request(path):
    response = server._process_request(SimpleNamespace(remote_address=("127.0.0.1", 0)),
                                      SimpleNamespace(headers={}, path=path))
    return json.loads(response.body)


def test_local_failure_does_not_change_provider(monkeypatch):
    monkeypatch.setenv("DEMON_ENHANCER_PROVIDER", "local")
    monkeypatch.setattr(pe, "_local_enhance", lambda *a: "")
    monkeypatch.setattr(pe, "_ask_haiku", lambda *a: pytest.fail("Must not call hosted"))
    assert pe.enhance_prompt("solo piano", "sa3", "local") == ("solo piano", False)


@pytest.mark.parametrize("provider", ["local", "LOCAL"])
def test_opted_out_local_provider_is_not_silently_substituted(monkeypatch, provider):
    monkeypatch.setenv("DEMON_ENHANCER_PROVIDER", "hosted")
    monkeypatch.setattr(pe, "enhance_prompt", lambda *a: pytest.fail("Must not invoke another provider"))
    assert not request("/api/enhance?prompt=solo+piano&provider=" + provider)["ok"]


@pytest.mark.parametrize("endpoint", ["enhance", "variations"])
def test_stale_revision_never_generates(monkeypatch, endpoint):
    monkeypatch.setenv("DEMON_ENHANCER_PROVIDER", "local")
    monkeypatch.setattr(pv, "identity", lambda: {"ok": True, "revision": "new"})
    monkeypatch.setattr(pv, "point", lambda *a, **k: pytest.fail("Stale variation ran"))
    monkeypatch.setattr(pe, "enhance_prompt", lambda *a, **k: pytest.fail("Stale enhancement ran"))
    result = request(f"/api/{endpoint}?prompt=solo+piano&revision=old")
    assert result["stale"] and not result["ok"] and result["revision"] == "new"


def test_identity_and_result_advertise_same_revision(monkeypatch):
    monkeypatch.setenv("DEMON_ENHANCER_PROVIDER", "local")
    monkeypatch.setattr(pv, "identity", lambda: {"ok": True, "revision": "new"})
    monkeypatch.setattr(pv, "point", lambda *a, **k: "solo piano")
    info = request("/api/prompt-model")
    result = request("/api/variations?prompt=solo+piano&revision=new")
    assert info["revision"] == result["revision"] == "new"
    assert result["ok"]


def test_revision_changes_for_weights_tokenizer_decoder_and_guard(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights-a")
    (tmp_path / "tokenizer.json").write_text("tokenizer-a")
    first = checkpoint_digest(str(tmp_path))
    assert first == checkpoint_digest(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"weights-b")
    second = checkpoint_digest(str(tmp_path))
    assert first != second
    (tmp_path / "tokenizer.json").write_text("tokenizer-b")
    assert second != checkpoint_digest(str(tmp_path))
    baseline = revision_for(first, "decoder1", "guard1", {"torch": "a"})
    assert baseline != revision_for(first, "decoder2", "guard1", {"torch": "a"})
    assert baseline != revision_for(first, "decoder1", "guard2", {"torch": "a"})
    assert baseline != revision_for(first, "decoder1", "guard1", {"torch": "b"})
