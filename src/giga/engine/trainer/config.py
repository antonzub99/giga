from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


@dataclass
class InstantiableConfig:
    """Base class for configs that can be instantiated from string."""

    _target_: str
    _partial_: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def instantiate(cls: Type[T], config: DictConfig) -> Any:
        """Instantiate an object from config with support for both nested configs and kwargs."""
        if not isinstance(config, DictConfig):
            return config

        if "_target_" not in config:
            return config

        kwargs = {k: v for k, v in config.items() if not k.startswith("_") and k != "kwargs"}
        if "kwargs" in config:
            kwargs.update(config.kwargs)

        module_path, class_name = config._target_.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        target_cls = getattr(module, class_name)

        # Recursively instantiate nested configs
        for key, value in kwargs.items():
            if isinstance(value, DictConfig):
                if "_target_" in value:
                    kwargs[key] = cls.instantiate(value)
                elif isinstance(value, (dict, DictConfig)):
                    kwargs[key] = {k: cls.instantiate(v) if isinstance(v, DictConfig) else v for k, v in value.items()}

        if config.get("_partial_", False):
            from functools import partial

            return partial(target_cls, **kwargs)

        return target_cls(**kwargs)


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    max_steps: Optional[int] = None
    device: str = "cuda"
    num_devices: int = 1
    ddp: bool = False
    mixed_precision: bool = True
    precision: Optional[str] = "bf16"
    compile: bool = False
    gradient_accumulation_steps: Optional[int] = 1
    scale_multiplier: float = 0.05
    bg_color: str = "black"
    texture_resolution: tuple[int, int] = (512, 512)
    texture_dropout: float = 0.0
    projection_settings: dict[str, Any] = field(default_factory=lambda: {"strategy": "softmax", "top_k": 4})
    num_warmup_steps: int = 5000


@dataclass
class LoggingConfig:
    """Logging configuration."""

    output_dir: str = "outputs/training"
    log_freq: int = 100
    log_stats_freq: int = 100
    log_media_freq: int = 1000
    save_freq: int = 5000
    eval_freq: int = 1000
    log_level: str = "INFO"


@dataclass
class DataConfig:
    """Data loading configuration."""

    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    persistent_workers: bool = True
    drop_last: bool = True


@dataclass
class MainConfig:
    """Main configuration combining all sub-configs."""

    experiment_name: str = "default_giga"
    pl_model: str = "monogiga.engine.models.GIGA"  # Default model class
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: InstantiableConfig = field(default_factory=dict)
    conditioner: InstantiableConfig = field(default_factory=dict)
    optimizer: InstantiableConfig = field(default_factory=dict)
    scheduler: Optional[InstantiableConfig] = None
    identity_conditioner: Optional[InstantiableConfig] = None
    loss: InstantiableConfig = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: str | Path, cli_overrides: list[str] | None = None) -> "MainConfig":
        """Load configuration from YAML file and merge with CLI overrides."""
        base_config = OmegaConf.load(config_path)

        if cli_overrides:
            cli_config = OmegaConf.from_dotlist(cli_overrides)
            base_config = OmegaConf.merge(base_config, cli_config)

        return cls(**base_config)

    def save(self, config_path: str | Path) -> None:
        """Save configuration to YAML file."""
        OmegaConf.save(self, config_path)
