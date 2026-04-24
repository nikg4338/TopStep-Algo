from __future__ import annotations

from pathlib import Path

import pytest

from replay_debug import _run_exit_simulator_for_artifact


@pytest.mark.parametrize("artifact_kind", ["directory", "file"])
def test_run_exit_simulator_for_artifact_uses_session_directory(monkeypatch, tmp_path: Path, artifact_kind: str) -> None:
    session_dir = tmp_path / "session_20260218"
    session_dir.mkdir(parents=True)
    if artifact_kind == "directory":
        artifact_path = session_dir
    else:
        artifact_path = session_dir / "session_summary.json"
        artifact_path.write_text("{}", encoding="utf-8")

    recorded: dict[str, str] = {}

    class FakeMRExitSimulator:
        def simulate_from_replay_artifacts(
            self,
            *,
            session_dir: str,
            replay_start: str,
            replay_end: str,
            symbol: str,
        ):
            recorded["session_dir"] = session_dir
            recorded["replay_start"] = replay_start
            recorded["replay_end"] = replay_end
            recorded["symbol"] = symbol
            return [object(), object()]

    monkeypatch.setattr("simulation.mr_exit_simulator.MRExitSimulator", FakeMRExitSimulator)

    trades_emitted = _run_exit_simulator_for_artifact(
        artifact_path,
        replay_start="2026-02-18T14:30:00Z",
        replay_end="2026-02-18T21:00:00Z",
        symbol="MES.c.0",
    )

    assert trades_emitted == 2
    assert recorded == {
        "session_dir": str(session_dir),
        "replay_start": "2026-02-18T14:30:00Z",
        "replay_end": "2026-02-18T21:00:00Z",
        "symbol": "MES.c.0",
    }