import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torchmetrics
import torchtext.datasets as datasets
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from models import build_transformer


def get_all_sentences(ds, lang):
    """
    Generator that yields all sentences from a specified language in a dataset.

    Args:
        ds: A dataset containing multilingual translation pairs. 
        lang (str): A string specifying the language code to extract sentences for.

    Yields:
        A sentence (str) from the dataset corresponding to the specified language.
    """
    for item in ds:
        yield item['translation'][lang]

