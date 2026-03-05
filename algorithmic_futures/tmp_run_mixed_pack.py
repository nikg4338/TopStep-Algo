from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

from validation.validation_pack import SessionEntry, ValidationPack, ValidationPackRunner


sessions: list[SessionEntry] = [
    SessionEntry("flat_low_vol_session", "2026-02-18T14:30:00Z", "2026-02-18T18:30:00Z", "range"),
    SessionEntry("runaway_trend_day", "2026-02-19T14:30:00Z", "2026-02-19T18:30:00Z", "trend"),
    SessionEntry("high_vol_news_day", "2026-02-20T14:30:00Z", "2026-02-20T18:30:00Z", "event"),
]

day_specs = [
    ("2026-02-18", "range"),
    ("2026-02-19", "trend"),
    ("2026-02-20", "chop"),
]
window_starts = [
    "14:30", "14:45", "15:00", "15:15", "15:30",
    "15:45", "16:00", "16:15", "16:30",
]


def add_minutes(day: str, hhmm: str, minutes: int = 90) -> tuple[str, str]:
    start = datetime.strptime(f"{day}T{hhmm}:00Z", "%Y-%m-%dT%H:%M:%SZ")
    end = start + timedelta(minutes=minutes)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


for day, cat in day_specs:
    for idx, hhmm in enumerate(window_starts, 1):
        s, e = add_minutes(day, hhmm, 90)
        sessions.append(SessionEntry(f"{cat}_{day.replace('-', '')}_{idx:02d}", s, e, cat))

pack = ValidationPack(
    pack_id="mixed_regimes_30_v1",
    description="30-session mixed-regime pack with explicit flat/trend/news anchors",
    sessions=sessions,
)

runner = ValidationPackRunner(pack, artifacts_root="artifacts/validation_runs", continue_on_error=True)
manifest = runner.run()
run_dir = Path("artifacts/validation_runs") / manifest.run_id

base = json.loads((run_dir / "mc_results.json").read_text())
mild = json.loads((run_dir / "mc_results_stress_mild.json").read_text())
severe = json.loads((run_dir / "mc_results_stress_severe.json").read_text())

scenarios = ["Base", "Mild", "Severe"]
p_target = [
    base.get("p_target_before_ruin", 0.0),
    mild.get("p_target_before_ruin", 0.0),
    severe.get("p_target_before_ruin", 0.0),
]
p_ruin = [
    base.get("p_ruin", 0.0),
    mild.get("p_ruin", 0.0),
    severe.get("p_ruin", 0.0),
]

x = range(len(scenarios))
width = 0.38
plt.figure(figsize=(9, 5.2))
plt.bar([i - width / 2 for i in x], p_target, width=width, label="p_target_before_ruin")
plt.bar([i + width / 2 for i in x], p_ruin, width=width, label="p_ruin")
plt.xticks(list(x), scenarios)
plt.ylim(0, 1)
plt.ylabel("Probability")
plt.title("Monte Carlo Survival: Base vs Mild vs Severe Stress")
plt.legend()
plt.tight_layout()
plot_path = run_dir / "stress_survival_comparison.png"
plt.savefig(plot_path, dpi=150)
plt.close()

print(f"RUN_ID={manifest.run_id}")
print(f"RUN_DIR={run_dir}")
print(f"P_TARGET_BASE={base.get('p_target_before_ruin')}")
print(f"P_TARGET_MILD={mild.get('p_target_before_ruin')}")
print(f"P_TARGET_SEVERE={severe.get('p_target_before_ruin')}")
print(f"P_RUIN_BASE={base.get('p_ruin')}")
print(f"P_RUIN_MILD={mild.get('p_ruin')}")
print(f"P_RUIN_SEVERE={severe.get('p_ruin')}")
print(f"PLOT_PATH={plot_path}")
