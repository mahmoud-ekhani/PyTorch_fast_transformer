from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class Config:
    batch_size: int = 8
    num_epochs: int = 20
    lr: float = 1e-4
    seq_len: int = 350
    embed_dim: int = 512
    src_lang: str = 'en'
    tgt_lang: str = 'it'
    data_source: str = 'opus_books'
    model_folder: str = 'checkpoints'
    model_basename: str = 'tmodel_'
    preload: str = 'latest'
    tokenizer_file: str = 'tokenizer_{0}.json'
    experiment_name: str = 'runs/tmodel'

def get_weights_file_path(config: Config, epoch: str) -> str:
    model_folder = Path('.') / f"{config.data_source}_{config.model_folder}"
    model_filename = f"{config.model_basename}{epoch}.pt"
    return str(model_folder / model_filename)

def latest_weights_file_path(config: Config) -> Optional[str]:
    model_folder = Path('.') / f"{config.data_source}_{config.model_folder}"
    model_filename = f"{config.model_basename}*"
    all_weights_files = list(model_folder.glob(model_filename))
    if not all_weights_files:
        return None
    all_weights_files.sort()
    return str(all_weights_files[-1])

