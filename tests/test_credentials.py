import json
import os


def test_local_credentials_are_literal_and_environment_takes_priority(tmp_path, monkeypatch):
    from jev_navi.credentials import load_local_env

    for name in ("BAMBOO_API_KEY", "STARTLUX_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STARTLUX_ENDPOINT", "https://existing.example/v1/decisions")
    key = "test-$HOME-$(literal)-secret"
    path = tmp_path / ".env"
    path.write_text(
        "BAMBOO_API_KEY=" + json.dumps(key) + "\n"
        "STARTLUX_ENDPOINT=https://file.example/v1/decisions\n"
        "UNRELATED=value\n"
    )
    load_local_env(path)
    assert os.environ["BAMBOO_API_KEY"] == key
    assert os.environ["STARTLUX_ENDPOINT"] == "https://existing.example/v1/decisions"
    assert "UNRELATED" not in os.environ


def test_saved_credentials_are_private_and_not_printed(tmp_path, capsys):
    from jev_navi.credentials import save_bamboo_credentials

    path = tmp_path / ".env"
    save_bamboo_credentials(path, "test-secret")
    assert path.stat().st_mode & 0o777 == 0o600
    assert "test-secret" not in capsys.readouterr().out
