import torch
import torch.nn as nn
import math
# from tokenize_bb import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import csv
from itertools import repeat

class TransformerModel(nn.Module):
    def __init__(self, output_dim=1000, d_model=100, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=100):
        super(TransformerModel, self).__init__()

        # Embedding layers
        # self.src_embedding = nn.Embedding(input_dim, d_model)
        # self.tgt_embedding = nn.Embedding(output_dim, d_model)

        # Positional encoding
        # self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Transformer model
        self.transformer = nn.Transformer(d_model, 10, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src, tgt):
        # Embed and apply positional encoding to source and target
        
        # src = self.src_embedding(src) * math.sqrt(self.d_model)
        # src = self.positional_encoding(src)

        # tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        # tgt = self.positional_encoding(tgt)

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

def train_loop(model, opt, loss_fn, dataloader, device=torch.device('cuda:0')): 
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        X, y = batch
        print(y)
        # X, y = batch[:, 0], batch[:, 1]
        X, y = [X], [y]
        X, y = torch.tensor(X).to(device), torch.tensor(y).to(device)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:, :-1]
        y_expected = y[:, 1:]
        
        # Get mask to mask out the next words
        sequence_length = len(y_input)
        # tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(X, y_input)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader, device=torch.device('cuda:0')):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            # X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.long, device=device), torch.tensor(y, dtype=torch.long, device=device)

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
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
        
    return total_loss / len(dataloader)

def train(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        
        # validation_loss = validation_loop(model, loss_fn, val_dataloader)
        # validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        # print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list

# # Prepare a dataset
# basic_blocks = [bb1, bb2, bb3]  # List of BasicBlock objects
# dataset = RegisterAllocationDataset(basic_blocks, tokenizer)

class RegisterAllocationDataset(Dataset):
    def __init__(self):
        # self.inputs = pd.read_csv("vectorized.csv", header=None)
        # self.labels = pd.read_csv("labels.csv", header=None)
        self.batch_size = 4
        with open("vectorized.csv", mode="r") as file:
            reader = csv.reader(file)
            self.inputs = [list(map(float, row)) for row in reader]
        
        with open("labels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [list(map(float, row)) for row in reader]
        
        self.data = [[inp, [[l] + [0] * 99 for l in label]] for inp, label in zip(self.inputs, self.labels)]

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
        return self.data[idx]

if __name__ == "__main__":
    model = TransformerModel()
    model = model.to(torch.device("cuda"))
    trainLoader = RegisterAllocationDataset()
    valLoader = RegisterAllocationDataset()
    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    train(model, optim, loss, trainLoader, valLoader, epochs=1)