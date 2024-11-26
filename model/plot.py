import matplotlib.pyplot as plt

epochs = 50
train_loss = []
val_loss = []
with open("output.txt", "r") as f:
    lines = [line for line in f]
    for line in lines:
        if "loss" in line:
            _, loss = line.strip().split(":")
            loss = float(loss.strip())
            if "Training" in line:
                train_loss.append(loss)
            else:
                val_loss.append(loss)

plt.plot(range(epochs), train_loss, label="Train")
plt.plot(range(epochs), val_loss, label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss.png")