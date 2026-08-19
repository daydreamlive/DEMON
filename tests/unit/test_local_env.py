import os

from demos.realtime_motion_graph_web.local_env import load_repo_env_defaults


def test_load_repo_env_defaults_sets_missing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
# comment
ANTHROPIC_API_KEY=local-key
ENHANCER_MODEL='local-model'
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ENHANCER_MODEL", raising=False)

    loaded = load_repo_env_defaults(env_file)

    assert loaded == 2
    assert os.environ["ANTHROPIC_API_KEY"] == "local-key"
    assert os.environ["ENHANCER_MODEL"] == "local-model"


def test_load_repo_env_defaults_preserves_exported_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=local-key\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported-key")

    loaded = load_repo_env_defaults(env_file)

    assert loaded == 0
    assert os.environ["ANTHROPIC_API_KEY"] == "exported-key"


def test_load_repo_env_defaults_ignores_missing_file(tmp_path):
    assert load_repo_env_defaults(tmp_path / "missing.env") == 0
