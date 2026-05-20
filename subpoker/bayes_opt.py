"""
Bayesian optimization for training Kuhn poker agents.
"""

import argparse
import json
from pathlib import Path
import sys


# Ensure the top-level ``bayes_opt`` package resolves before this compatibility
# wrapper when the file is executed directly from inside ``subpoker/``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
PACKAGE_ROOT = PROJECT_ROOT / "bayes_opt"
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)
if PACKAGE_ROOT.is_dir():
    __path__ = [str(PACKAGE_ROOT)]


from bayes_opt.config import (
    BAYES_DATA_ROOT,
    DEFAULT_FIRST_PLAYERS,
    DEFAULT_TEST_SEEDS,
    FIXED_NASH_ALPHA,
    TestingConfig,
    TrainingConfig,
    best_final_dir,
    default_study_name,
    metadata_from_training_config,
    seed_for_trial,
    storage_url,
    study_dir,
    training_config_from_params,
    trial_dir,
)

from bayes_opt.evaluation import evaluate_run
from bayes_opt.optimize import (
    HIDDEN_SIZE_CHOICES,
    build_parser as build_optimize_parser,
    optimize,
    suggest_params,
    suggest_training_config,
    write_best_trial,
    write_trials_csv,
)

from bayes_opt.train_best import train_best
from bayes_opt.training import TrainingResult, train_policy

__all__ = [
    "BAYES_DATA_ROOT",
    "DEFAULT_FIRST_PLAYERS",
    "DEFAULT_TEST_SEEDS",
    "FIXED_NASH_ALPHA",
    "HIDDEN_SIZE_CHOICES",
    "TestingConfig",
    "TrainingConfig",
    "TrainingResult",
    "best_final_dir",
    "build_parser",
    "evaluate_run",
    "main",
    "metadata_from_training_config",
    "optimize",
    "seed_for_trial",
    "storage_url",
    "study_dir",
    "suggest_params",
    "suggest_training_config",
    "train_best",
    "train_policy",
    "training_config_from_params",
    "trial_dir",
    "write_best_trial",
    "write_trials_csv",
]


def build_parser() -> argparse.ArgumentParser:
    parser = build_optimize_parser()
    parser.description = "Optimize NumNet hyperparameters, then train the best trial for a final long run."
    parser.add_argument("--final-episodes", type=int, default=1_000_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.study_name:
        args.study_name = default_study_name()

    study = optimize(args)
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage_url(args.study_name)}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    final_result = train_best(
        argparse.Namespace(
            study_name=args.study_name,
            episodes=args.final_episodes,
            test_episodes=args.test_episodes,
            seed=args.seed,
            no_progress=args.no_progress,
        )
    )
    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()
