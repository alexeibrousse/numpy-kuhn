"""Run Optuna hyperparameter optimization for the NumNet poker bot."""


import argparse
import json
from pathlib import Path


import optuna
from optuna.trial import FrozenTrial, TrialState

from bayes_opt.config import (
    TestingConfig,
    TrainingConfig,
    default_study_name,
    seed_for_trial,
    storage_url,
    study_dir,
    training_config_from_params,
    trial_dir,
)
from bayes_opt.evaluation import evaluate_run
from bayes_opt.training import train_policy


HIDDEN_SIZE_CHOICES = [8, 12, 16, 20, 32, 48, 64, 96, 128]


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 5e-4, log=True),
        "hidden_size": trial.suggest_categorical("hidden_size", HIDDEN_SIZE_CHOICES),
        "adam_beta1": trial.suggest_float("adam_beta1", 0.80, 0.99),
        "adam_beta2": trial.suggest_float("adam_beta2", 0.95, 0.9999),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "use_learning_rate_decay": trial.suggest_categorical("use_learning_rate_decay", [True, False]),
        "baseline_momentum": trial.suggest_float("baseline_momentum", 0.01, 0.50),
        "use_baseline_bound": trial.suggest_categorical("use_baseline_bound", [True, False]),
        "baseline_bound": trial.suggest_int("baseline_bound", 2, 30),
        "use_entropy_bonus": trial.suggest_categorical("use_entropy_bonus", [True, False]),
        "initial_entropy_coeff": trial.suggest_float("initial_entropy_coeff", 1e-4, 1e-1, log=True),
        "use_entropy_decay": trial.suggest_categorical("use_entropy_decay", [True, False]),
        "gradient_clip": trial.suggest_float("gradient_clip", 0.5, 25.0, log=True),
    }


def suggest_training_config(trial: optuna.Trial, episodes: int, seed: int) -> TrainingConfig:
    params = suggest_params(trial)
    return training_config_from_params(params, episodes=episodes, seed=seed)


def write_trials_csv(study: optuna.Study, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)


def write_best_trial(study: optuna.Study, directory: Path) -> None:
    try:
        best = study.best_trial
    except ValueError:
        return

    payload = {
        "number": best.number,
        "value": best.value,
        "params": best.params,
        "user_attrs": best.user_attrs,
    }
    with open(directory / "best_trial.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def optimize(args: argparse.Namespace) -> optuna.Study:
    directory = study_dir(args.study_name)
    directory.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url(args.study_name),
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    testing_config = TestingConfig(total_episodes=args.test_episodes)

    def objective(trial: optuna.Trial) -> float:
        run_dir = trial_dir(args.study_name, trial.number)
        trial_seed = seed_for_trial(args.seed, trial.number)
        training_config = suggest_training_config(trial, args.trial_episodes, trial_seed)

        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("random_seed", trial_seed)
        trial.set_user_attr("training_episodes", args.trial_episodes)
        trial.set_user_attr("test_episodes", args.test_episodes)

        training_result = train_policy(
            training_config,
            run_dir,
            save_full_training=False,
            run_analysis=False,
            show_progress=not args.no_progress,
        )
        test_summary = evaluate_run(
            run_dir,
            testing_config,
            write_full_data=False,
            write_plot=False,
            show_progress=not args.no_progress,
        )

        objective_value = float(test_summary["average_reward"])
        trial.set_user_attr("training_summary", training_result.summary)
        trial.set_user_attr("testing_summary", test_summary)
        return objective_value

    def after_trial(study: optuna.Study, trial: FrozenTrial) -> None:
        write_trials_csv(study, directory)
        write_best_trial(study, directory)

    complete_trials = sum(1 for trial in study.trials if trial.state == TrialState.COMPLETE)
    remaining_trials = max(0, args.n_trials - complete_trials)
    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=args.n_jobs,
            callbacks=[after_trial],
            show_progress_bar=False,
        )
    write_trials_csv(study, directory)
    write_best_trial(study, directory)
    return study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize NumNet hyperparameters with Optuna.")
    parser.add_argument("--study-name")
    parser.add_argument("--n-trials", type=int, default=75)
    parser.add_argument("--trial-episodes", type=int, default=300000)
    parser.add_argument("--test-episodes", type=int, default=168000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--no-progress", action="store_true")
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


if __name__ == "__main__":
    main()
