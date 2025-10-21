import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random


############################
# 1. Signal Generation
############################

def generate_time_series_dataset(O, F, P, T):
    """
    Generate a temporal signal of O dimensions, T timepoints, with frequencies F and powers P.
    Each dimension can have its own powers in the P matrix (shape O x len(F))
    """
    t = np.arange(T)
    sig = np.zeros((O, T), dtype=np.float32)
    F = np.array(F)
    P = np.array(P) # shape: O x len(F)
    phases = 2 * np.pi * np.random.rand(O, len(F))
    for o in range(O):
        for k, freq in enumerate(F):
            sig[o] += np.sqrt(P[o, k]) * np.sin(2 * np.pi * freq * t / T + phases[o, k])
    return sig.T  # shape (T, O)


############################
# 2. Dataset Creation
############################

def build_dataset(O, F_list, P_list, T, S, sigma, split=(0.8, 0.2), seed=42):
    """
    Generates 5 classes (A-E), S samples per class, temporal signals of O dims, length T
    F_list, P_list: lists of frequencies and per-class powers
    sigma: magnitude of Gaussian noise added to each sample
    Returns: train_loader, test_loader, input_dim, num_classes
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    classes = ['A','B','C','D','E']
    num_classes = len(classes)
    all_X = []
    all_Y = []
    for c in range(num_classes):
        F = F_list[c]
        P = P_list[c]
        for s in range(S):
            sig = generate_time_series_dataset(O, F, P, T)  # shape (T, O)
            sig += np.random.normal(0, sigma, sig.shape)    # add noise
            all_X.append(sig)
            all_Y.append(c)
    all_X = np.stack(all_X)   # [5S, T, O]
    all_Y = np.array(all_Y)   # [5S]
    # Shuffle
    idx = np.arange(len(all_Y))
    np.random.shuffle(idx)
    all_X = all_X[idx]
    all_Y = all_Y[idx]
    # Split
    n_train = int(len(all_Y)*split[0])
    X_train, Y_train = all_X[:n_train], all_Y[:n_train]
    X_test, Y_test = all_X[n_train:], all_Y[n_train:]
    # Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.long)
    # DataLoaders
    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    return train_loader, test_loader, O, num_classes

############################
# 3. PyTorch Vanilla RNN
############################

class RNNClassifierReconstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1,
                          nonlinearity=nonlinearity, batch_first=True)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.recon_head = nn.Linear(hidden_dim, input_dim)  # output same shape as input

    def forward(self, x):
        # x: [batch, seq, input_dim]
        out, h_n = self.rnn(x)  # out: [batch, seq, hidden_dim]
        last_hidden = out[:, -1, :]   # [batch, hidden_dim]
        class_logits = self.class_head(last_hidden)
        recon_out = self.recon_head(out)  # [batch, seq, input_dim]
        return class_logits, recon_out, out

############################
# 4&5. Train/Test with toggles
############################

def train_model(model, train_loader, test_loader, epochs=10, 
                alpha=1.0, use_mse=True, use_ce=True, 
                reconstruction_time=0, device='cpu'):
    """
    reconstruction_time:
        0 = input_t ~ output_t (reconstruct present input)
       <0 = input_{t+abs(k)} ~ output_t (reconstruct past input, k negative)
       >0 = input_{t-k} ~ output_t (predict future input, k positive)
    alpha: scaling for MSE loss
    """
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_accs, train_losses, test_accs, test_losses = [], [], [], []

    for ep in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            logits, recon, _ = model(X_batch)
            # CE Loss
            loss = 0
            if use_ce:
                loss_ce = ce_loss(logits, Y_batch)
                loss += loss_ce
            # MSE Loss (reconstruction)
            if use_mse:
                shift = reconstruction_time
                # Make recon target and output match in time
                # X_batch: [B, T, O], recon: [B, T, O]
                if shift == 0:
                    target = X_batch
                    pred = recon
                elif shift < 0:
                    pad = torch.zeros_like(X_batch[:, :abs(shift), :])
                    target = torch.cat([pad, X_batch[:, :X_batch.shape[1]+shift, :]], dim=1)
                    pred = recon
                elif shift > 0:
                    pad = torch.zeros_like(X_batch[:, -shift:, :])
                    target = torch.cat([X_batch[:, shift:, :], pad], dim=1)
                    pred = recon
                loss_mse = mse_loss(pred, target)
                loss += alpha * loss_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            pred_class = logits.argmax(dim=1)
            correct += (pred_class == Y_batch).sum().item()
            total += X_batch.size(0)
        train_losses.append(total_loss / total)
        train_accs.append(correct / total)
        # Evaluate
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                logits, recon, _ = model(X_batch)
                loss = 0
                if use_ce:
                    loss_ce = ce_loss(logits, Y_batch)
                    loss += loss_ce
                if use_mse:
                    shift = reconstruction_time
                    if shift == 0:
                        target = X_batch
                        pred = recon
                    elif shift < 0:
                        pad = torch.zeros_like(X_batch[:, :abs(shift), :])
                        target = torch.cat([pad, X_batch[:, :X_batch.shape[1]+shift, :]], dim=1)
                        pred = recon
                    elif shift > 0:
                        pad = torch.zeros_like(X_batch[:, -shift:, :])
                        target = torch.cat([X_batch[:, shift:, :], pad], dim=1)
                        pred = recon
                    loss_mse = mse_loss(pred, target)
                    loss += alpha * loss_mse
                pred_class = logits.argmax(dim=1)
                correct += (pred_class == Y_batch).sum().item()
                total += X_batch.size(0)
                total_loss += loss.item() * X_batch.size(0)
        test_losses.append(total_loss / total)
        test_accs.append(correct / total)
        print(f'Epoch {ep+1} Train Acc: {train_accs[-1]:.3f} Test Acc: {test_accs[-1]:.3f} ' +
              f'Train Loss: {train_losses[-1]:.3f} Test Loss: {test_losses[-1]:.3f}')
    return train_accs, train_losses, test_accs, test_losses

############################
# 6. Plotting
############################

def plot_metrics(train_accs, train_losses, test_accs, test_losses):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_accs, 'o-', label='Train Acc')
    plt.plot(test_accs, 'o-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_losses, 'o-', label='Train Loss')
    plt.plot(test_losses, 'o-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

############################
# 7. Example Usage
############################

if __name__ == "__main__":
    # Settings
    O = 2
    T = 1000
    S = 20 # samples per class
    sigma = 0.3

    # Five sets of frequencies and powers for five classes: shape (class, O x len(F))
    F_list = [
        [2,8,16],
        [2,6,12],
        [3,9,15],
        [1,5,14],
        [4,10,18]
    ]
    # Powers: each class gets matrix shape [O, len(F)]
    P_list = [
        np.array([[5,1,1],[2,2,6]]), 
        np.array([[4,3,1],[1,5,2]]),
        np.array([[7,2,2],[2,1,7]]),
        np.array([[3,4,2],[4,2,3]]),
        np.array([[2,2,6],[6,1,2]])
    ]
    # Create dataset
    train_loader, test_loader, input_dim, num_classes = build_dataset(
        O, F_list, P_list, T, S, sigma, split=(0.8, 0.2))
    # Model
    hidden_dim = 128
    model = RNNClassifierReconstructor(input_dim, hidden_dim, num_classes).to('cpu')
    # Choose options
    use_ce = True
    use_mse = True
    alpha = 1.0 # set to 0 to ignore MSE
    reconstruction_time = 0  # 0 = present, -1 = previous, +1 = future
    # Train
    train_accs, train_losses, test_accs, test_losses = train_model(
        model, train_loader, test_loader,
        epochs=10, alpha=alpha, use_mse=use_mse, use_ce=use_ce, 
        reconstruction_time=reconstruction_time
    )
    # Plot
    plot_metrics(train_accs, train_losses, test_accs, test_losses)