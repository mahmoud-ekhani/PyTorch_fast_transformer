import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self,
                 ds,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_lang: str,
                 tgt_lang: str,
                 seq_len: int):
        """
        Args:
            ds: Dataset.
            src_tokenizer:
            tgt_tokenizer:
            src_lang: The source language.
            tgt_lang: The target language.
            seq_len: The sequence length.
        """
        super(BilingualDataset, self).__init__()
        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token_id = torch.tensor([tgt_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token_id = torch.tensor([tgt_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token_id = torch.tensor([tgt_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        """
        Returns:
            Length of the dataset.
        """
        return len(self.ds)
    
    def _causal_mask(self, size):
        """
        Args:
            size: Size of the causal mask tensor.
        
        Returns: A lower triangular mask tensor to prevent attending to future tokens.
        """
        mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
        return mask == 0
    
    def __getitem__(self, idx):
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        # Transform text into token ids
        encoder_input_token_ids = self.src_tokenizer.encode(src_text).ids
        decoder_input_token_ids = self.tgt_tokenizer.encode(tgt_text).ids

        # Calculate the number of padding tokens
        num_encoder_padding_tokens = self.seq_len - len(encoder_input_token_ids) - 2 # We will add SOS and EOS
        num_decoder_padding_tokens = self.seq_len - len(decoder_input_token_ids) - 1 # We only add SOS

        # Ensure number of the padding tokens is not negative (i.e., the sentence is not too long)
        assert num_encoder_padding_tokens >= 0; "Source sequence is too long."
        assert num_decoder_padding_tokens >= 0; "Target sequence is too long."

        # Add SOS, EOS, and PADDING token ids to the source sequence
        encoder_input = torch.cat([
            self.sos_token_id,
            torch.tensor(encoder_input_token_ids, dtype=torch.int64),
            self.eos_token_id,
            torch.tensor([self.pad_token_id] * num_encoder_padding_tokens, dtype=torch.int64)
        ],
        dim=0
        )

        # Add SOS and PADDING token ids to the target sequence
        decoder_input = torch.cat([
            self.sos_token_id,
            torch.tensor(decoder_input_token_ids, dtype=torch.int64),
            torch.tensor([self.pad_token_id] * num_decoder_padding_tokens, dtype=torch.int64)
        ],
        dim=0
        )

        # Add EOS and PADDING token ids to the target sequence to create the labels tensor
        label = torch.cat([
            torch.tensor(decoder_input_token_ids, dtype=torch.int64),
            self.eos_token_id,
            torch.tensor([self.pad_token_id] * num_decoder_padding_tokens, dtype=torch.int64)
        ],
        dim=0
        )

        # Confirm all the tensors are seq_len long
        assert encoder_input.size(0) == self.seq_len; "Length of the encoder input is incorrect"
        assert decoder_input.size(0) == self.seq_len; "Length of the decoder input is incorrect"
        assert label.size(0) == self.seq_len; "Length of the labels is incorrect"

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(1).int(), # [1, 1, seq_len]
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & self._causal_mask(decoder_input.size(0)), # [1, seq_len] & [1, seq_len, seq_len]
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }   
