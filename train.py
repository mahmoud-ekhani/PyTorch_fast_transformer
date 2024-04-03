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

def run_validate(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, 
                 global_step, writer, num_examples=2, console_width=80):
    """
    Runs the validation process for the given model on a validation dataset.

    Args:
        model: The trained model to be validated.
        validation_ds (DataLoader): A PyTorch DataLoader containing the validation dataset.
        tokenizer_src (Tokenizer): The tokenizer for the source language.
        tokenizer_tgt (Tokenizer): The tokenizer for the target language.
        max_len (int): The maximum length of the sequences to be generated.
        device: The device on which to perform the computations.
        print_msg (function): A function to print messages.
        global_step (int): The current global step in the training process.
        writer (SummaryWriter): A Tensorboard SummaryWriter instance.
        num_examples (int, optional): The number of examples to validate. Defaults to 2.
        console_width (int, optional): The width of the console for printing. Defaults to 80.

    The function validates the model by decoding a fixed number of examples from the 
    validation dataset and computing various metrics like character error rate, word error rate, 
    and BLEU score.
    """
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            if encoder_input.size(0) != 1:
                raise ValueError("Batch size of 1 is required for validation")

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            print_validation_results(print_msg, console_width, source_text, target_text, model_out_text)

            if count == num_examples:
                break

    if writer:
        calculate_and_log_metrics(writer, global_step, predicted, expected)

def print_validation_results(print_msg, console_width, source_text, target_text, model_out_text):
    """
    Helper function to print the validation results for each example.

    Args:
        print_msg (function): Function to print messages.
        console_width (int): Width of the console.
        source_text (str): The source text.
        target_text (str): The expected target text.
        model_out_text (str): The predicted text by the model.
    """
    print_msg('-'*console_width)
    print_msg(f"{f'SOURCE: ':>12}{source_text}")
    print_msg(f"{f'TARGET: ':>12}{target_text}")
    print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
    print_msg('-'*console_width)

def calculate_and_log_metrics(writer, global_step, predicted, expected):
    """
    Helper function to calculate and log metrics like character error rate, word error rate,
    and BLEU score using the torchmetrics library.

    Args:
        writer (SummaryWriter): Tensorboard SummaryWriter instance.
        global_step (int): The current global step in the training process.
        predicted (list of str): The list of predicted sentences.
        expected (list of str): The list of expected (ground truth) sentences.
    """
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    writer.add_scalar('validation cer', cer, global_step)

    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    writer.add_scalar('validation wer', wer, global_step)

    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    writer.add_scalar('validation BLEU', bleu, global_step)
    writer.flush()
    
def train_model(config):
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    if (device == "cuda"):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index) / 1024 ** 3} GB")
    elif (device == "mps"):
        print(f"Device name: <mps>")
    else:
        print("Note: No GPU was found.")
    device = torch.device(device)

    # Confirm the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Load a user pre-specified (if any) mdel
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch.')
        initial_epoch = 0
        global_step = 0
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Running epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask) # Shape: [batch_size, seq_len, embed_dim]
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # Shape: [batch_size, seq_len, embed_dim]
            proj_output = model.project(decoder_output) # Shape: [batch_size, seq_len, vocab_size]

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

            run_validate(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)