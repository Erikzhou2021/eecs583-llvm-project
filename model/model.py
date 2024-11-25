import torch
import torch.nn as nn
import math
# from tokenize_bb import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import csv
from itertools import repeat
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


torch.manual_seed(583)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
# NUM_PHYS_REG = 1000
NUM_TOKENS = 25000
# NUM_TOKENS = 1000
# NUM_TOKENS = 1000
MAX_INPUT_LEN = 1000
OUTPUT_DIMS = 1000

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
        self.src_embedding = nn.Embedding(output_dim, d_model) # Just going to pretend we have 25000 different tokens, hopefully this works
        self.tgt_embedding = nn.Embedding(output_dim, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer model
        self.transformer = nn.Transformer(d_model, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, batch_first=True)

        # Output layer
        self.fc_out = nn.Linear(d_model, OUTPUT_DIMS)
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


def train_loop(model, opt, loss_fn, dataloader, scaler, scheduler): 
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        print(f"\rTrain batch {i + 1}/{len(dataloader)}", end="")
        X, y = batch
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_input = y_input.masked_fill(y_input == -1, 0)
            y_expected = y[:, 1:]
            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            src_pad_mask = model.create_pad_mask(X, 0).to(device)
            tgt_pad_mask = model.create_pad_mask(y_input, 0).to(device)

            pred = model(X, y_input, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
            pred = pred.reshape(-1, OUTPUT_DIMS)
            y_expected = y_expected.reshape(-1)
            y_expected = y_expected.type(torch.LongTensor)
            
            # pred = pred.masked_fill(pred >= 4000, -1)
            y_expected = y_expected.masked_fill(y_expected >= OUTPUT_DIMS, -1)
            # y_expected = y_expected.masked_fill(y_expected >= 4000, -1)
            
            loss = loss_fn(pred, y_expected.to(device))

        opt.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scheduler.step()
        scaler.update()
    
        total_loss += loss.detach().item()
    print() 
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader): 
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"\rValidation batch {i + 1}/{len(dataloader)}", end="")
            # opt.zero_grad()
            X, y = batch
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                y_input = y[:, :-1]
                y_input = y_input.masked_fill(y_input == -1, 0)
                y_expected = y[:, 1:]
                # Get mask to mask out the next words
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(device)
                src_pad_mask = model.create_pad_mask(X, 0).to(device)
                tgt_pad_mask = model.create_pad_mask(y_input, 0).to(device)

                pred = model(X, y_input, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
                pred = pred.reshape(-1, OUTPUT_DIMS)
                y_expected = y_expected.reshape(-1)
                y_expected = y_expected.type(torch.LongTensor)
                
                # pred = pred.masked_fill(pred >= 4000, -1)
                y_expected = y_expected.masked_fill(y_expected >= OUTPUT_DIMS, -1)
                # y_expected = y_expected.masked_fill(y_expected >= 4000, -1)
                
                loss = loss_fn(pred, y_expected.to(device))

            # scaler.scale(loss).backward()
            # scaler.step(opt)
            # scaler.update()
        
            total_loss += loss.detach().item()
    print() 
    return total_loss / len(dataloader)


def testing(model, loss_fn, dataloader):
    model.eval()
    # total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"\rbatch {i + 1}/{len(dataloader)}", end="")
            X, y = batch
            # X, y = batch[:, 0], batch[:, 1]
            # X, y = torch.tensor(X, dtype=torch.int32, device=device), torch.tensor(y, dtype=torch.int32, device=device)
            X, y = X.to(device), y.to(device)

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
            # break
    print()
    # return total_acc
    return total_acc / len(dataloader)

def train(model, opt, loss_fn, train_dataloader, val_dataloader, scaler, scheduler, epochs):
    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []
    max_patience = 10
    curr_patience = 0
    min_val_loss = 9999999999
    
    print("Training and validating model")
    if epochs != -1:
        for epoch in range(epochs):
            start = time.time()
            print("-"*25, f"Epoch {epoch + 1}","-"*25)
            print(f"lr = {scheduler.get_last_lr()[0]:.6f}")
            
            train_loss = train_loop(model, opt, loss_fn, train_dataloader, scaler, scheduler)
            train_loss_list += [train_loss]
            print(f"Training loss: {train_loss:.4f}")
            print(f"{(time.time() - start):.02f} s")
            
            start = time.time()
            validation_loss = validation_loop(model, loss_fn, val_dataloader)
            validation_loss_list += [validation_loss]
            print(f"Validation loss: {validation_loss:.4f}")
            print(f"{(time.time() - start):.02f} s")
            print()
            torch.save(model.state_dict(), "weights_checkpoint.pt")
        num_epochs = epochs
    else: # patience
        num_epochs = 0
        while curr_patience < max_patience:
            start = time.time()
            print("-"*25, f"Epoch {num_epochs + 1}","-"*25)
            print(f"lr = {scheduler.get_last_lr()[0]:.6f}")
            
            train_loss = train_loop(model, opt, loss_fn, train_dataloader, scaler, scheduler)
            train_loss_list += [train_loss]
            print(f"Training loss: {train_loss:.4f}")
            print(f"{(time.time() - start):.02f} s")
            
            start = time.time()
            validation_loss = validation_loop(model, loss_fn, val_dataloader)
            validation_loss_list += [validation_loss]
            print(f"Validation loss: {validation_loss:.4f}")
            print(f"{(time.time() - start):.02f} s")
            print()
            if validation_loss < min_val_loss:
                torch.save(model.state_dict(), "weights_best_val_loss.pt")
                min_val_loss = validation_loss
                curr_patience = 0
            else:
                curr_patience += 1
            num_epochs += 1
        
    return train_loss_list, validation_loss_list, num_epochs

def inference(model, X):
    y = [] # gets passed into the model
    y_pred = [] # just the physical registers
    i = 0
    while i < len(X):
        if X[i] >= 3000:
            y.append(X[i].item())
        if len(y) > 1000:
            break
        # print(torch.min(torch.tensor(y)), torch.max(torch.tensor(y)))
        tgt_mask = model.get_tgt_mask(len(y)).to(device)
        pred = model(torch.unsqueeze(X, 0), torch.tensor([y], device=device), tgt_mask)
        pred = pred[0][-1]
        # print("pred", pred.shape, pred)
        predToken = torch.argmax(pred).item() # getting the predicted value
        # print("predToken", type(predToken), predToken)
        # predToken = pred.topk(1)[1].view(-1)[-1].item()
        if i == len(X)-1 or (i < len(X)-2 and X[i+1] < 3000):
            y.append(predToken)
            y_pred.append(predToken)
        # print("y", y)
        i += 1
    return y_pred

def calcAccuracy(y, y_expected):
    count = 0
    print()
    y, y_expected = y[0], y_expected[0]
    print("LENGTH", len(y), len(y_expected))
    print("y", y)
    print("y_expected", [i.item() for i in y_expected])
    for i in range(len(y_expected)):
        if y[i] == y_expected[i].item():
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


def warmup_only_lr_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        return 1.0  # Keep LR constant after warmup
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_inverse_sqrt_scheduler(optimizer, d_model, warmup_steps):
    def lr_lambda(step):
        scale = d_model ** -0.5
        return scale * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_scheduler(optimizer, d_model, warmup_steps, min_lr=1e-5):
    def lr_lambda(step):
        # Linear warmup phase
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        return 1.0
        # scale = d_model ** -0.5
        # lr_decay = scale * min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5)
        # # Ensure the learning rate does not go below the minimum threshold
        # return max(min_lr, lr_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
            # self.data = self.data[:128]

        with open("allDataLabels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            # self.labels = self.labels[:128]
        
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
        with open("allDataInput.csv", mode="r") as file:
            reader = csv.reader(file)
            self.data = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            # print("self.data", type(self.data), type(self.data[0]), type(self.data[0][0]))
            self.data = self.data[6524:]

        with open("allDataLabels.csv", mode="r") as file:
            reader = csv.reader(file)
            self.labels = [torch.tensor(list(map(float, row)), dtype=torch.int32) for row in reader]
            self.labels = self.labels[6524:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

if __name__ == "__main__":
    batch_size = 8
    epochs = -1

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    model = TransformerModel().to(device)
    # model.load_state_dict(torch.load("weights_8k.pt", weights_only=True))
    
    # load dataset
    dataset = RegisterAllocationDataset()
    # Define split sizes
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True)
    valData = ValDataset()
    valLoader = DataLoader(valData, batch_size=1, collate_fn=collate_fn)
    
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optim = torch.optim.AdamW(params=model.parameters(), lr=2e-4, weight_decay=1e-3)
    warmup = int(0.05 * (30 * train_size / batch_size))
    print(f"warmup steps: {warmup}")
    # scheduler = warmup_only_lr_scheduler(optim, warmup_steps=warmup)
    scheduler = get_scheduler(optim, 512, warmup)
    
    scaler = torch.amp.GradScaler()
    train_loss, val_loss, epochs = train(model, optim, loss, train_loader, val_loader, scaler, scheduler, epochs=epochs)
    
    plt.plot(range(epochs), train_loss, label="Train")
    plt.plot(range(epochs), val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss.png")
    # torch.save(model.state_dict(), "weights.pt")
    
    # model.load_state_dict(torch.load("weights_8k.pt", weights_only=True))
    print(testing(model, None, valLoader))