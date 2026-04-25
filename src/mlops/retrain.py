"""
Scheduled retraining workflow with approval gate and rollback marker.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


def _load_cfg(root: Path) -> dict:
    return yaml.safe_load((root / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    root = Path(__file__).parents[2]
    cfg = _load_cfg(root)
    models_dir = root / cfg["data"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    previous_champion = models_dir / "champion_metrics.json"
    backup_path = models_dir / "champion_metrics.backup.json"
    if previous_champion.exists():
        backup_path.write_text(previous_champion.read_text(encoding="utf-8"), encoding="utf-8")

    _run(["python", "-m", "src.models.train_xgboost"], root)
    _run(["python", "-m", "src.models.train_lstm"], root)
    _run(["python", "-m", "src.models.evaluate"], root)

    eval_path = root / "reports" / "evaluation_summary.json"
    if not eval_path.exists():
        raise RuntimeError("Evaluation summary missing; retraining aborted")

    summary = json.loads(eval_path.read_text(encoding="utf-8"))
    new_f1 = float(summary.get("classifier", {}).get("f1_macro", 0.0))

    old_f1 = 0.0
    if previous_champion.exists():
        old = json.loads(previous_champion.read_text(encoding="utf-8"))
        old_f1 = float(old.get("f1_macro", 0.0))

    min_gain = float(cfg.get("mlops", {}).get("champion_challenger", {}).get("promote_if_f1_gain_min", 0.005))
    approved = (new_f1 - old_f1) >= min_gain

    approval = {
        "new_f1": new_f1,
        "old_f1": old_f1,
        "required_gain": min_gain,
        "approved": approved,
    }
    (models_dir / "retrain_approval.json").write_text(json.dumps(approval, indent=2), encoding="utf-8")

    if not approved and backup_path.exists():
        previous_champion.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
