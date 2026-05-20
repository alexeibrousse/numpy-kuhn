"""Configuration objects and path helpers for Optuna bot optimization."""

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BAYES_DATA_ROOT = PROJECT_ROOT / "data" / "bayes_opt"
FIXED_NASH_ALPHA = 1.0 / 3.0
MAX_RANDOM_SEED = 2**32 - 1
RUN_NAME_FORMAT = "%d-%m-%y_%H-%M"

DEFAULT_TEST_SEEDS = (
    1337,
    271828,
    314159,
    8675309,
    29461070,
    52960675432,
    381420072889,
)
DEFAULT_FIRST_PLAYERS = (0, 1)


@dataclass
class TrainingConfig:
    n_epochs: int = 500_000
    learning_rate: float = 5e-5
    hidden_size: int = 20
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    use_learning_rate_decay: bool = True
    use_baseline: bool = True
    baseline_momentum: float = 0.10
    use_baseline_bound: bool = True
    baseline_bound: float = 15.0
    use_entropy_bonus: bool = False
    initial_entropy_coeff: float = 0.01
    use_entropy_decay: bool = True
    use_gradient_clipping: bool = True
    gradient_clip: float = 10.0
    nash_alpha: float = FIXED_NASH_ALPHA
    random_seed: int = 0
    player_number: int = 0
    input_size: int = 12
    output_size: int = 4


@dataclass
class TestingConfig:
    total_episodes: int = 168000
    seeds: tuple[int, ...] = DEFAULT_TEST_SEEDS
    first_players: tuple[int, ...] = DEFAULT_FIRST_PLAYERS
    nash_alpha: float = FIXED_NASH_ALPHA
    player_number: int = 0


def default_study_name() -> str:
    return datetime.now().strftime(RUN_NAME_FORMAT)


def latest_study_name() -> str:
    if not BAYES_DATA_ROOT.exists():
        raise FileNotFoundError(f"No Bayes optimization runs found in {BAYES_DATA_ROOT}.")

    study_dirs = [path for path in BAYES_DATA_ROOT.iterdir() if path.is_dir()]
    if not study_dirs:
        raise FileNotFoundError(f"No Bayes optimization runs found in {BAYES_DATA_ROOT}.")

    return max(study_dirs, key=lambda path: path.stat().st_mtime).name


def study_dir(study_name: str) -> Path:
    return BAYES_DATA_ROOT / study_name


def trial_dir(study_name: str, trial_number: int) -> Path:
    return study_dir(study_name) / f"trial_{trial_number:04d}"


def best_final_dir(study_name: str) -> Path:
    return study_dir(study_name) / "best_final"


def storage_url(study_name: str) -> str:
    return f"sqlite:///{study_dir(study_name) / 'study.db'}"


def seed_for_trial(base_seed: int, trial_number: int) -> int:
    return (base_seed + trial_number * 9_973) % MAX_RANDOM_SEED


def metadata_from_training_config(config: TrainingConfig) -> dict:
    return {
        "implementation": "numpy",
        "agent": f"NashAgent(alpha={config.nash_alpha:.3f})",
        "optimizer": "adam",
        "input_size": config.input_size,
        "hidden_size": config.hidden_size,
        "output_size": config.output_size,
        "initial_learning_rate": config.learning_rate,
        "adam_beta1": config.adam_beta1,
        "adam_beta2": config.adam_beta2,
        "adam_epsilon": config.adam_epsilon,
        "use_learning_rate_decay": config.use_learning_rate_decay,
        "activation": "ReLU",
        "number_epochs": config.n_epochs,
        "use_baseline": config.use_baseline,
        "baseline_momentum": config.baseline_momentum,
        "use_baseline_bound": config.use_baseline_bound,
        "baseline_bound": config.baseline_bound,
        "use_entropy_bonus": config.use_entropy_bonus,
        "initial_entropy_coeff": config.initial_entropy_coeff,
        "use_entropy_decay": config.use_entropy_decay,
        "random_seed": config.random_seed,
        "use_gradient_clipping": config.use_gradient_clipping,
        "gradient_clip": config.gradient_clip,
        "nash_alpha": config.nash_alpha,
        "player_number": config.player_number,
        "optuna_config": dataclass_to_dict(config),
    }


def training_config_from_params(params: dict, episodes: int, seed: int,) -> TrainingConfig:
    use_entropy_bonus = bool(params["use_entropy_bonus"])
    return TrainingConfig(
        n_epochs=episodes,
        learning_rate=float(params["learning_rate"]),
        hidden_size=int(params["hidden_size"]),
        adam_beta1=float(params["adam_beta1"]),
        adam_beta2=float(params["adam_beta2"]),
        adam_epsilon=float(params.get("adam_epsilon", TrainingConfig.adam_epsilon)),
        use_learning_rate_decay=bool(params["use_learning_rate_decay"]),
        baseline_momentum=float(params["baseline_momentum"]),
        use_baseline_bound=bool(params["use_baseline_bound"]),
        baseline_bound=float(params["baseline_bound"]),
        use_entropy_bonus=use_entropy_bonus,
        initial_entropy_coeff=float(params.get("initial_entropy_coeff", TrainingConfig.initial_entropy_coeff)),
        use_entropy_decay=bool(params.get("use_entropy_decay", TrainingConfig.use_entropy_decay)),
        gradient_clip=float(params["gradient_clip"]),
        random_seed=seed,
    )


def dataclass_to_dict(obj):
    """Recursively convert a dataclass (or nested dataclasses) to a plain dict.

    This replaces `dataclasses.asdict` to avoid the dependency and keep control
    over the exact conversion behavior.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(value)
        return result
    if isinstance(obj, dict):
        return type(obj)((dataclass_to_dict(k), dataclass_to_dict(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        converted = tuple(dataclass_to_dict(v) for v in obj)
        return type(obj)(converted) if isinstance(obj, list) else converted
    return obj
