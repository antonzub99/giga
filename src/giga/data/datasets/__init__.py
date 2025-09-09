# Import all dataset modules to trigger registration
from . import dna, mvh, mvhpp, neuman
from .config import (
    DatasetConfig,
    instantiate_dataset,
    load_dataset_config,
    register_dataset,
    select_characters,
)

# Re-export dataset classes for backward compatibility
from .dna import DNADataset
from .mvh import MVHAposeDataset, MVHDataset
from .mvhpp import MVHPPAposeDataset, MVHPPDataset
from .neuman import NeumanDataset
from .utils.split_manager import DatasetSplitManager, SplitConfig
