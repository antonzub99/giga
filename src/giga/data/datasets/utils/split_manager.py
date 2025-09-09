from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from loguru import logger


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""

    method: str = "character"  # 'character', 'timestep', or 'index'
    character_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    character_indices: dict[str, tuple[int, int]] = field(default_factory=dict)
    timestep_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    seed: int = 42

    @classmethod
    def character_based(
        cls,
        train_range: tuple[str, str],
        val_range: tuple[str, str],
        test_range: tuple[str, str],
        seed: int = 42,
    ) -> "SplitConfig":
        """Create config for character-based splitting."""
        return cls(
            method="character",
            character_ranges={
                "train": train_range,
                "val": val_range,
                "test": test_range,
            },
            seed=seed,
        )

    @classmethod
    def timestep_based(
        cls,
        train_range: tuple[float, float] = (0.0, 0.7),
        val_range: tuple[float, float] = (0.7, 0.85),
        test_range: tuple[float, float] = (0.85, 1.0),
        seed: int = 42,
    ) -> "SplitConfig":
        """Create config for timestep-based splitting."""
        return cls(
            method="timestep",
            timestep_ranges={
                "train": train_range,
                "val": val_range,
                "test": test_range,
            },
            seed=seed,
        )

    @classmethod
    def index_based(
        cls,
        train_range: tuple[int, int] = (0, 2500),
        val_range: tuple[int, int] = (2500, 3000),
        test_range: tuple[int, int] | None = None,
        seed: int = 42,
    ) -> "SplitConfig":
        """Create config for index-based character splitting."""
        indices = {
            "train": train_range,
            "val": val_range,
        }
        if test_range:
            indices["test"] = test_range

        return cls(
            method="index",
            character_indices=indices,
            seed=seed,
        )


class DatasetSplitManager:
    """Manages dataset splitting strategy."""

    def __init__(self, config: SplitConfig):
        self.config = config
        np.random.seed(config.seed)

    def get_split_items(self, characters: list[str], split: str, timesteps: Sequence[int] | None = None) -> tuple[list[str], list[int]]:
        """Get items for specified split.

        Args:
            characters: List of all available character names
            split: Split name ('train', 'val', or 'test')
            timesteps: Optional list of timesteps

        Returns:
            Tuple of (character_names, timesteps) for the requested split
        """
        if self.config.method == "character":
            return self._split_by_characters(characters, split), timesteps
        elif self.config.method == "index":
            return self._split_by_indices(characters, split), timesteps
        else:
            return characters, self._split_by_timesteps(timesteps, split)

    def _split_by_characters(self, characters: list[str], split: str) -> list[str]:
        """Split dataset by character names."""
        if split not in self.config.character_ranges:
            raise ValueError(f"Split '{split}' not found in character_ranges.")

        start_name, end_name = self.config.character_ranges[split]
        sorted_chars = sorted(characters)

        try:
            start_idx = sorted_chars.index(start_name)
        except ValueError:
            start_idx = next(
                (i for i, name in enumerate(sorted_chars) if name >= start_name),
                len(sorted_chars),
            )

        try:
            end_idx = sorted_chars.index(end_name) + 1
        except ValueError:
            end_idx = next(
                (i for i, name in enumerate(sorted_chars) if name > end_name),
                len(sorted_chars),
            )

        split_chars = sorted_chars[start_idx:end_idx]
        logger.info(f"{split} split: {len(split_chars)} characters")
        return split_chars

    def _split_by_timesteps(self, timesteps: Sequence[int], split: str) -> list[int]:
        """Split dataset by timesteps."""
        if timesteps is None:
            raise ValueError("Timesteps must be provided for timestep-based splitting")

        if split not in self.config.timestep_ranges:
            raise ValueError(f"Split '{split}' not found in timestep_ranges.")

        start_ratio, end_ratio = self.config.timestep_ranges[split]
        total_steps = len(timesteps)

        start_idx = int(total_steps * start_ratio)
        end_idx = int(total_steps * end_ratio)

        split_steps = list(timesteps[start_idx:end_idx])
        logger.info(f"{split} split: {len(split_steps)} timesteps")
        return split_steps

    def _split_by_indices(self, characters: list[str], split: str) -> list[str]:
        """Split dataset by character indices."""
        if split not in self.config.character_indices:
            raise ValueError(f"Split '{split}' not found in character_indices.")

        start_idx, end_idx = self.config.character_indices[split]
        sorted_chars = sorted(characters)

        # Clamp indices to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(sorted_chars), end_idx)

        split_chars = sorted_chars[start_idx:end_idx]
        logger.info(f"{split} split: {len(split_chars)} characters (indices {start_idx}:{end_idx})")
        return split_chars
