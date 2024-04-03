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

def get_or_build_tokenizer(config, ds, lang):
    """
    Retrieves or builds a tokenizer for the specified language.

    This function checks if a pre-trained tokenizer for the given language exists. 
    Otherwise, a new tokenizer is trained using the WordLevel model on 
    the sentences extracted from the provided dataset.

    Args:
        config (dict): A configuration dictionary.
        ds: The dataset used for training the tokenizer if it doesn't already exist.
        lang (str): The language code for which the tokenizer is to be retrieved or built.

    Returns:
        A Tokenizer object for the specified language. The tokenizer is either loaded 
        from a pre-existing file or trained and then saved to a file.

    Note:
        The tokenizer training uses the 'get_all_sentences' function to extract sentences 
        from the dataset for the specified language.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))

    if not Path.exists(tokenizer_path):
        # Create a new tokenizer and train it
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                                   min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        # Load an existing tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_ds(config):
    """
    Loads a dataset for the specified language pair, splits it into 
    training and validation sets, builds or loads tokenizers for each language, and
    wraps the datasets into torch DataLoader objects for efficient batching during training.
    """
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90/10 splitting for training and validation sets
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, 
                                tokenizer_src, 
                                tokenizer_tgt, 
                                config['lang_src'],
                                config['lang_tgt'],
                                config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw,
                              tokenizer_src,
                              tokenizer_tgt,
                              config['lang_src'],
                              config['lang_tgt'],
                              config['seq_len'])
    
    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of the source setences: {max_len_src}')
    print(f'Max length of the target setences: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config["seq_len"], embed_dim=config['d_model'])
    return model

def greedy_decode(model, src, src_mask, src_tokenizer, tgt_tokenizer, max_len, device):
    """
    Performs greedy decoding for a sequence-to-sequence model.

    This function takes an input sequence, processes it through the model, and attempts to 
    generate the target sequence one token at a time. Decoding is performed greedily, meaning 
    at each step, it chooses the token with the highest probability from the model's output.

    Args:
        model: Transformer model, with `encode` and `decode` methods.
        src (torch.Tensor): The input tensor for the source sequence of shape [batch_size, src_seq_len].
        src_mask (torch.Tensor): The mask tensor for the source sequence of shape [batch_size, 1, 1, src_seq_len].
        src_tokenizer (Tokenizer): The tokenizer used for the source language.
        tgt_tokenizer (Tokenizer): The tokenizer used for the target language.
        max_len (int): The maximum length of the generated sequence.
        device: The device on which the tensors should be processed. 

    Returns:
        torch.Tensor: The tensor containing the token IDs of the generated sequence, of shape 
                      [tgt_seq_len]. The sequence starts with the SOS token and ends with the 
                      EOS token or when `max_len` is reached.

    The function continues to generate tokens until the EOS token is generated or the length 
    of the generated sequence reaches `max_len`. 
    """
    sos_idx = tgt_tokenizer.token_to_id('[SOS]')
    eos_idx = tgt_tokenizer.token_to_id('[EOS]')

    # Precompute the encoder output, and reuse it for every step
    encoder_output = model.encode(src, src_mask)

    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(src_mask).to(device)

        # Calculate the output
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).fill_(next_word.item()).type_as(src).to(device)],
            dim=1
        )

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)

