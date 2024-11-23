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
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
# NUM_PHYS_REG = 1000
NUM_TOKENS = 25000
# NUM_TOKENS = 1000
MAX_INPUT_LEN = 1000

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
    def __init__(self, output_dim=NUM_TOKENS, d_model=512, max_len=MAX_INPUT_LEN):
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

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
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
        memory = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        # memory = self.transformer(src, tgt)

        # Output layer
        output = self.fc_out(memory)
        return output

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


def train_loop(model, opt, loss_fn, dataloader, scaler): 
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        print(f"\rbatch {i + 1}", end="")
        opt.zero_grad()
        X, y = batch
        # print(X.shape)
        # print(y.shape)
        # X, y = batch[:, 0], batch[:, 1]
        # X, y = [X], [y]
        # X, y = torch.tensor(X, dtype=torch.int32).to(device), torch.tensor(y, dtype=torch.int32).to(device)
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
        # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_input = y_input.masked_fill(y_input == -1, 0)
            y_expected = y[:, 1:]
            # print(y.device, y_input.device, y_expected.device)
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            src_pad_mask = model.create_pad_mask(X, 0).to(device)
            tgt_pad_mask = model.create_pad_mask(y_input, 0).to(device)

            # print(X.shape, y.shape)
            # Standard training except we pass in y_input and tgt_mask
            # pred = model(X, y_input, tgt_mask)
            # pred = model(X, y_input)
            pred = model(X, y_input, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)

            # Permute pred to have batch size first again
            # print(pred.shape)
            # pred = pred.permute(1, 2, 0) 
            
            # pred = pred.contiguous().view(-1, NUM_TOKENS)
            # y_expected = y_expected.contiguous().view(-1)
            pred = pred.reshape(-1, NUM_TOKENS)
            y_expected = y_expected.reshape(-1)
            y_expected = y_expected.type(torch.LongTensor)
            
            # print(pred.shape, pred.dtype)  
            # print(y_expected.shape, y_expected.dtype)  
            # print(pred.shape)
            # loss = loss_fn(pred, y_expected)
            loss = loss_fn(pred, y_expected.to(device))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # loss.backward()
        # opt.step()
    
        total_loss += loss.detach().item()
    print() 
    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader):
    model.eval()
    # total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            # X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X, dtype=torch.int32, device=device), torch.tensor(y, dtype=torch.int32, device=device)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            # y_expected = y[:, 1:]
            
            # Get mask to mask out the next words
            # sequence_length = len(y_input)
            # tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            # pred = model(X, y_input)
            pred = []
            for row in X:
                pred.append(inference(model, row))

            # Permute pred to have batch size first again
            # pred = pred.permute(1, 2, 0)  
            # pred = pred.contiguous().view(-1, NUM_TOKENS)
            # y_expected = y_expected.contiguous().view(-1)
            # y_expected = y_expected.type(torch.LongTensor) 

            y_expected = []
            for row in y:
                temp = []
                for val in row:
                    if val < 1000:
                        temp.append(val)
                y_expected.append(temp)

            total_acc += calcAccuracy(pred, y_expected)
            # loss = loss_fn(pred, y_expected.to(device))
            # # loss = loss_fn(pred, y_expected.to(device))
            # total_loss += loss.detach().item()
        
    return total_acc / len(dataloader)

def train(model, opt, loss_fn, train_dataloader, val_dataloader, scaler, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    
    print("Training and validating model")
    for epoch in range(epochs):
        start = time.time()
        print("-"*25, f"Epoch {epoch + 1}","-"*25)
        
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, scaler)
        train_loss_list += [train_loss]
        
        # validation_loss = validation_loop(model, loss_fn, val_dataloader)
        # validation_loss_list += [validation_loss]
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"{(time.time() - start):.02f} s")
        # print(f"Validation loss: {validation_loss:.4f}")
        print()
        
    return train_loss_list, validation_loss_list

def inference(model, X):
    y = []
    y_pred = []
    i = 0
    while i < len(X) - 1:
        if X[i] >= 3000:
            y.append(X[i])
        if len(y) > 1000:
            break
        pred = model(torch.unsqueeze(X, 0), torch.tensor([y]))
        predToken = torch.argmax(pred[-1]).item()
        if i == len(X)-1 or (i < len(X)-2 and X[i+1] < 3000):
            y.append(predToken)
            y_pred.append(predToken)
    return y

def calcAccuracy(y, y_expected):
    count = 0
    for i in range(len(y_expected)):
        if y[i] == y_expected:
            count += 1
    return count / len(y_expected)

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unpack inputs and labels
    # print(type(batch))
    # print(type(sequences), type(labels))
    # print(type(sequences[0]))
    sequences = [sequence[:MAX_INPUT_LEN] for sequence in sequences]
    labels = [label[:MAX_INPUT_LEN] for label in labels]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Use a different padding value for labels
    return padded_sequences, padded_labels

# def collate_fn(batch):
#     sequences, labels = zip(*batch)
#     sequences = [torch.tensor(sequence[:MAX_INPUT_LEN], dtype=torch.int32) for sequence in sequences]
#     labels = [torch.tensor(label[:MAX_INPUT_LEN], dtype=torch.int32) for label in labels]
#     padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
#     padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
#     return padded_sequences, padded_labels


class RegisterAllocationDataset(Dataset):
    def __init__(self):
        # self.inputs = pd.read_csv("vectorized.csv", header=None)
        # self.labels = pd.read_csv("labels.csv", header=None)
        # with open("vectorized.csv", mode="r") as file:
        #     reader = csv.reader(file)
        #     self.inputs = [list(map(float, row)) for row in reader]
        with open("allDataInput.csv", mode="r") as file:
            reader = csv.reader(file)
            self.data = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            # self.data = self.data[:1024]

        with open("allDataLabels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            # self.labels = self.labels[:1024]
        
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

class ValDataset(Dataset):
    def __init__(self):
        with open("input.csv", mode="r") as file:
            reader = csv.reader(file)
            self.data = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            self.data = self.data

        with open("labels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            self.labels = self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    torch.cuda.empty_cache()
    model = TransformerModel().to(device)
    # model.load_state_dict(torch.load("weights_1024.pt", weights_only=True))
    torch.backends.cudnn.benchmark = True
    trainData = RegisterAllocationDataset()
    valData = ValDataset()
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optim = torch.optim.Adam(params=model.parameters(), lr=2e-4)
    # print(sum(p.numel() for p in model.parameters()))
    # exit(0)
    trainLoader = DataLoader(trainData, batch_size=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    valLoader = DataLoader(valData, batch_size=1, collate_fn=collate_fn, pin_memory=True)
    scaler = torch.amp.GradScaler()
    print(validation_loop(model, None, valLoader))
    # train(model, optim, loss, trainLoader, trainLoader, scaler, epochs=10)
    torch.save(model.state_dict(), "weights.pt")