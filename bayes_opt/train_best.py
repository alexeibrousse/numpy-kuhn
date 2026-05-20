"""Train and evaluate the best Optuna configuration for a longer final run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import optuna

from bayes_opt.config import (
    TestingConfig,
    best_final_dir,
    latest_study_name,
    seed_for_trial,
    storage_url,
    study_dir,
    training_config_from_params,
)

from bayes_opt.evaluation import evaluate_run
from bayes_opt.training import train_policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_best(args: argparse.Namespace) -> dict:
    if not args.study_name:
        try:
            args.study_name = latest_study_name()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    directory = study_dir(args.study_name)
    directory.mkdir(parents=True, exist_ok=True)

    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage_url(args.study_name),
        )
        best_trial = study.best_trial
    except KeyError as exc:
        raise SystemExit(f"Study not found: {args.study_name}") from exc
    except ValueError as exc:
        raise SystemExit(f"Study has no completed trials: {args.study_name}") from exc

    final_seed = seed_for_trial(args.seed, best_trial.number + 1_000_000)
    training_config = training_config_from_params(
        best_trial.params,
        episodes=args.episodes,
        seed=final_seed,
    )
    run_dir = best_final_dir(args.study_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
                "final_random_seed": final_seed,
            },
            f,
            indent=4,
        )

    training_result = train_policy(
        training_config,
        run_dir,
        save_full_training=True,
        run_analysis=False,
        show_progress=not args.no_progress,
    )
    analysis_script = PROJECT_ROOT / "subpoker" / "train_analysis.py"
    subprocess.run([sys.executable, str(analysis_script), str(run_dir)], check=True)
    testing_summary = evaluate_run(
        run_dir,
        TestingConfig(total_episodes=args.test_episodes),
        write_full_data=True,
        write_plot=True,
        show_progress=not args.no_progress,
    )

    result = {
        "run_dir": str(run_dir),
        "best_trial_number": best_trial.number,
        "best_trial_value": best_trial.value,
        "training_summary": training_result.summary,
        "testing_summary": testing_summary,
    }
    with open(run_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the best Optuna trial for a final long run.")
    parser.add_argument("--study-name")
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--test-episodes", type=int, default=168_000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = train_best(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
