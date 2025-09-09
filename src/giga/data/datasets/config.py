from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from loguru import logger
from omegaconf import OmegaConf

from .utils.split_manager import DatasetSplitManager, SplitConfig

T = TypeVar("T")

# Registry for dataset classes and their configs
DATASET_REGISTRY = {}


def register_dataset(name: str):
    """Decorator to register a dataset class with its config class."""

    def decorator(cls):
        if hasattr(cls, "__config_class__"):
            DATASET_REGISTRY[name] = {"dataset": cls, "config": cls.__config_class__}
        return cls

    return decorator


def get_dataset_info(name: str):
    """Get dataset class and config class by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def _extract_dataset_name(target: str) -> str:
    """Extract dataset name from _target_ field."""
    class_name = target.split(".")[-1]  # Get the actual class name
    # Convert class name to lowercase and remove "Dataset" suffix
    dataset_name = class_name.lower().replace("dataset", "")
    return dataset_name


@dataclass
class DatasetConfig:
    """Base configuration for all datasets."""

    data_dir: Path
    _target_: str
    num_input_cameras: list[int] = field(default_factory=lambda: [4])
    num_target_cameras: list[int] = field(default_factory=lambda: [4])
    image_size: tuple[int, int] = (1024, 1024)
    texture_resolution: tuple[int, int] = (512, 512)
    timesteps: list[int] = field(default_factory=lambda: [-1])
    patch_config: dict[str, Any] = field(default_factory=lambda: {"num_patches": 8, "patch_size": 256, "patch_scale": [0.5, 1.0, 2.0]})
    character_selection: list[str | int] = field(default_factory=lambda: ["all"])
    exclusions_file: Optional[Path] = None
    split_config: Optional[SplitConfig] = None
    texture_dir: Optional[Path] = None

    @classmethod
    def load(cls: Type[T], config_path: str | Path) -> T:
        """Load configuration from YAML file."""
        config = OmegaConf.load(config_path)
        config.data_dir = Path(config.data_dir)
        if "exclusions_file" in config and config.exclusions_file is not None:
            config.exclusions_file = Path(config.exclusions_file)
        if "texture_dir" in config and config.texture_dir is not None:
            config.texture_dir = Path(config.texture_dir)
        return cls(**config)


def select_characters(config: DatasetConfig) -> list[str]:
    """Select characters based on dataset type."""
    dataset_name = _extract_dataset_name(config._target_)
    dataset_info = get_dataset_info(dataset_name)

    # Get the character selection function from the dataset class
    dataset_cls = dataset_info["dataset"]
    if hasattr(dataset_cls, "select_characters"):
        return dataset_cls.select_characters(config)
    else:
        # Fallback to importing the function based on dataset name
        # Ideally should never happen
        from .utils.character_selection import (
            select_characters_dna,
            select_characters_mvh,
            select_characters_neuman,
        )

        selector_map = {
            "mvh": select_characters_mvh,
            "mvhpp": select_characters_mvh,
            "mvhppa": select_characters_mvh,
            "dna": select_characters_dna,
            "neuman": select_characters_neuman,
        }

        if dataset_name not in selector_map:
            raise ValueError(f"No character selector found for dataset: {dataset_name}")

        return selector_map[dataset_name](config.data_dir, config.character_selection, config.exclusions_file)


def instantiate_dataset(
    config: DatasetConfig,
    split: str = "train",
) -> Any:
    """Instantiate a dataset from config.

    Args:
        config: Dataset configuration
        split: Dataset split to use ("train", "val", or "test")

    Returns:
        Instantiated dataset
    """
    # Get dataset class from registry
    dataset_name = _extract_dataset_name(config._target_)
    dataset_info = get_dataset_info(dataset_name)
    dataset_cls = dataset_info["dataset"]

    characters = select_characters(config)
    timesteps = config.timesteps if hasattr(config, "timesteps") else None

    if config.split_config is not None:
        split_manager = DatasetSplitManager(config.split_config)
        characters, timesteps = split_manager.get_split_items(characters, split, timesteps)
        # Update timesteps in config if split by timesteps
        if split_manager.config.method == "timestep" and hasattr(config, "timesteps") and timesteps is not None:
            config.timesteps = timesteps
    elif split != "train":
        logger.warning(f"No split configuration provided but '{split}' split requested. Using all available data.")

    return dataset_cls(
        config=config,
        characters=characters,
        split=split,
    )


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    """Load and instantiate correct dataset config based on _target_ field."""
    base_config = OmegaConf.load(config_path)
    if "_target_" not in base_config:
        raise ValueError("Config must have '_target_' field")

    dataset_name = _extract_dataset_name(base_config._target_)
    dataset_info = get_dataset_info(dataset_name)
    config_class = dataset_info["config"]

    return config_class.load(config_path)
