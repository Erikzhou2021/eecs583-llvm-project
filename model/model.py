import torch
import torch.nn as nn
import math
from tokenize import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=100):
        super(TransformerModel, self).__init__()

        # Embedding layers
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_embedding = nn.Embedding(output_dim, d_model)

        # Positional encoding
        # self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Transformer model
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and apply positional encoding to source and target
        
        # src = self.src_embedding(src) * math.sqrt(self.d_model)
        # src = self.positional_encoding(src)

        # tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # tgt = self.positional_encoding(tgt)

        # Pass through the transformer
        memory = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        # Output layer
        output = self.fc_out(memory)
        return output

# Example usage
input_dim = 1000  # Vocabulary size for source
output_dim = 1000  # Vocabulary size for target
model = TransformerModel(input_dim, output_dim)



from torch.utils.data import Dataset

class RegisterAllocationDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv("vectorized.csv", header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the basic block and its corresponding register mappings
        # bb = self.basic_blocks[idx]
        # input_text = " ".join(bb.to_tokenized_format())  # Input sequence
        # output_text = get_output_mappings(bb)  # Output sequence (e.g., "{v0 -> p1, ...}")

        # # Tokenize the input and output sequences
        # inputs = self.tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        # outputs = self.tokenizer(output_text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)

        # # Create a dictionary for the model
        # return {
        #     "input_ids": inputs["input_ids"].squeeze(0),
        #     "attention_mask": inputs["attention_mask"].squeeze(0),
        #     "labels": outputs["input_ids"].squeeze(0),
        # }
        return self.data.iloc[idx]

# # Prepare a dataset
# basic_blocks = [bb1, bb2, bb3]  # List of BasicBlock objects
# dataset = RegisterAllocationDataset(basic_blocks, tokenizer)