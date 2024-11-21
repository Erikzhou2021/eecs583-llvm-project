import torch
import torch.nn as nn
import math
# from tokenize_bb import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import csv
from itertools import repeat
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# NUM_PHYS_REG = 1000
NUM_TOKENS = 25000
# NUM_TOKENS = 1000

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, output_dim=NUM_TOKENS, d_model=512, max_len=1000):
        super(TransformerModel, self).__init__()

        # Embedding layers
        self.src_embedding = nn.Embedding(NUM_TOKENS, d_model) # Just going to pretend we have 25000 different tokens, hopefully this works
        self.tgt_embedding = nn.Embedding(NUM_TOKENS, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer model
        self.transformer = nn.Transformer(d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(d_model, NUM_TOKENS)
        self.d_model = d_model

    def forward(self, src, tgt):
        # Embed and apply positional encoding to source and target
        
        # src = self.src_embedding(src) * math.sqrt(self.d_model)
        # print(torch.min(src), torch.max(src))
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        # print(src.shape)

        # tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # print(torch.min(tgt), torch.max(tgt))
        tgt = self.tgt_embedding(tgt)
        # print(tgt.shape)
        tgt = self.positional_encoding(tgt)
        # print(tgt.shape)
        # Pass through the transformer
        # src: our custom tokenized input
        # tgt: 1D list of physical reg, shifted by 1
        # tgt: [opcode physreg, physreg, ..., opcde2, physreg, physreg]
        # example: eax = add ebx ecx, xor edx eax ecx
        # src: MIR2Vec representation
        # training (tgt) ->  add, eax, ebx, ecx, xor, eax, ecx
        # tgt output -> eax, ebx, ecx, xor, eax, ecx


        # ground truth -> eax, ebx, ecx, eax, ecx

        
        # inference -> add, _predictedOut_, _predictedOut2_, _predictedOut3, xor, _predictedOut4_, _predictedOut5_, predictedOut6_
        memory = self.transformer(src, tgt)

        # Output layer
        output = self.fc_out(memory)
        return output

def train_loop(model, opt, loss_fn, dataloader): 
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        opt.zero_grad()
        X, y = batch
        # print(X.shape)
        # print(y.shape)
        # X, y = batch[:, 0], batch[:, 1]
        # X, y = [X], [y]
        X, y = torch.tensor(X, dtype=torch.int32).to(device), torch.tensor(y, dtype=torch.int32).to(device)
        # X, y = X.to(device), y.to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_input = y_input.masked_fill(y_input == -1, 0)
        y_expected = y[:, 1:]
        
        # Get mask to mask out the next words
        sequence_length = len(y_input)
        # tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # print(X.shape, y.shape)
        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input)

        # Permute pred to have batch size first again
        # print(pred.shape)
        # pred = pred.permute(1, 2, 0) 
        
        pred = pred.contiguous().view(-1, NUM_TOKENS)
        y_expected = y_expected.contiguous().view(-1)
        y_expected = y_expected.type(torch.LongTensor)
        # print(pred.shape, pred.dtype)  
        # print(y_expected.shape, y_expected.dtype)  
        # print(pred.shape)
        loss = loss_fn(pred, y_expected.to(device))

        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            # X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.int32, device=device), torch.tensor(y, dtype=torch.int32, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            
            # Get mask to mask out the next words
            sequence_length = len(y_input)
            # tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(X, y_input)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)  
            pred = pred.contiguous().view(-1, NUM_TOKENS)
            y_expected = y_expected.contiguous().view(-1)
            y_expected = y_expected.type(torch.LongTensor) 

            loss = loss_fn(pred, y_expected.to(device))
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def train(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        start = time.time()
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        # validation_loss = validation_loop(model, loss_fn, val_dataloader)
        # validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"{(time.time() - start):.02f} s")
        # print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list

# # Prepare a dataset
# basic_blocks = [bb1, bb2, bb3]  # List of BasicBlock objects
# dataset = RegisterAllocationDataset(basic_blocks, tokenizer)

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unpack inputs and labels
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Use a different padding value for labels
    return padded_sequences, padded_labels


class RegisterAllocationDataset(Dataset):
    def __init__(self):
        # self.inputs = pd.read_csv("vectorized.csv", header=None)
        # self.labels = pd.read_csv("labels.csv", header=None)
        self.batch_size = 4
        # with open("vectorized.csv", mode="r") as file:
        #     reader = csv.reader(file)
        #     self.inputs = [list(map(float, row)) for row in reader]
        const = 100
        with open("input.csv", mode="r") as file:
            reader = csv.reader(file)
            self.data = [torch.tensor(list(map(float, row))) for row in reader]
            self.data = self.data * const

        with open("labels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [torch.tensor(list(map(float, row))) for row in reader]
            self.labels = self.labels * const
        
        # for sequence in self.data:
        #     assert all(0 <= token < NUM_TOKENS for token in sequence), "Token index out of range!"
        # for sequence in self.labels:
        #     assert all(0 <= token < NUM_TOKENS for token in sequence), "Token index out of range!"
        # self.data = [[inp, label] for inp, label in zip(self.inputs, self.labels)]
        # self.data = [[inp, [[l] + [0] * 99 for l in label]] for inp, label in zip(self.inputs, self.labels)]

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
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    model = TransformerModel()
    model = model.to(device)
    trainData = RegisterAllocationDataset()
    valLoader = RegisterAllocationDataset()
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    # print(sum(p.numel() for p in model.parameters()))
    # exit(0)
    trainLoader = DataLoader(trainData, batch_size=16, shuffle=True, collate_fn=collate_fn)

    train(model, optim, loss, trainLoader, trainLoader, epochs=5)