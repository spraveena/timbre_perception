import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import copy
import random

############################
# Data Generation Utilities
############################

def generate_time_series_dataset(O, F, P, T):
    t = np.arange(T)
    sig = np.zeros((O, T), dtype=np.float32)
    F = np.array(F)
    P = np.array(P) # shape: O x len(F)
    phases = 2 * np.pi * np.random.rand(O, len(F))
    for o in range(O):
        for k, freq in enumerate(F):
            sig[o] += np.sqrt(P[o, k]) * np.sin(2 * np.pi * freq * t / T + phases[o, k])
    return sig.T  # shape (T, O)

def build_dataset(O, F_list, P_list, T, S, sigma, split=(0.8, 0.2), seed=42):
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
# Model
############################

class RNNClassifierReconstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1,
                          nonlinearity=nonlinearity, batch_first=True)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.recon_head = nn.Linear(hidden_dim, input_dim)  # output same shape as input

    def forward(self, x):
        out, h_n = self.rnn(x)  # out: [batch, seq, hidden_dim]
        last_hidden = out[:, -1, :]
        class_logits = self.class_head(last_hidden)
        recon_out = self.recon_head(out)  # [batch, seq, input_dim]
        return class_logits, recon_out, out

############################
# Pretraining
############################

def pretrain_reconstruction(model, train_loader, test_loader, epochs=10, reconstruction_time=0, device='cpu'):
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_losses, test_losses = [], []
    for ep in range(epochs):
        # Train
        model.train()
        total_loss = 0
        total = 0
        for X, Y in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            _, recon, _ = model(X)
            shift = reconstruction_time
            if shift == 0:
                target = X
            elif shift < 0:
                pad = torch.zeros_like(X[:, :abs(shift), :])
                target = torch.cat([pad, X[:, :X.shape[1]+shift, :]], dim=1)
            elif shift > 0:
                pad = torch.zeros_like(X[:, -shift:, :])
                target = torch.cat([X[:, shift:, :], pad], dim=1)
            loss = mse_loss(recon, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total += X.size(0)
        train_losses.append(total_loss / total)
        # Test
        model.eval()
        total_loss = 0
        total = 0
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(device)
                _, recon, _ = model(X)
                shift = reconstruction_time
                if shift == 0:
                    target = X
                elif shift < 0:
                    pad = torch.zeros_like(X[:, :abs(shift), :])
                    target = torch.cat([pad, X[:, :X.shape[1]+shift, :]], dim=1)
                elif shift > 0:
                    pad = torch.zeros_like(X[:, -shift:, :])
                    target = torch.cat([X[:, shift:, :], pad], dim=1)
                loss = mse_loss(recon, target)
                total_loss += loss.item() * X.size(0)
                total += X.size(0)
            test_losses.append(total_loss / total)
        print(f'Pretrain Epoch {ep+1}: Train Recon Loss: {train_losses[-1]:.4f} | Test Recon Loss: {test_losses[-1]:.4f}')
    return train_losses, test_losses

def plot_losses(train_losses, test_losses, title="Reconstruction loss"):
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()

############################
# Classification Training
############################

def train_classification(model, train_loader, test_loader, epochs=10, device='cpu'):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for ep in range(epochs):
        model.train()
        total = 0
        correct = 0
        total_loss = 0
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(X)
            loss = ce_loss(logits, Y)
            loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=1)
            total += X.size(0)
            correct += (pred == Y).sum().item()
            total_loss += loss.item() * X.size(0)
        train_accs.append(correct / total)
        train_losses.append(total_loss / total)
        # Test
        model.eval()
        total = 0
        correct = 0
        total_loss = 0
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)
                logits, _, _ = model(X)
                loss = ce_loss(logits, Y)
                pred = logits.argmax(dim=1)
                total += X.size(0)
                correct += (pred == Y).sum().item()
                total_loss += loss.item() * X.size(0)
            test_accs.append(correct / total)
            test_losses.append(total_loss / total)
        print(f'Classify Epoch {ep+1}: Train Acc: {train_accs[-1]:.3f} | Test Acc: {test_accs[-1]:.3f} ' +
              f'| Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}')
    return train_accs, test_accs, train_losses, test_losses

def plot_classification(train_accs, test_accs, train_losses, test_losses, title='Classification performance'):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_accs, 'o-', label='Train Acc')
    plt.plot(test_accs, 'o-', label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(title)
    plt.subplot(1,2,2)
    plt.plot(train_losses, 'o-', label='Train Loss')
    plt.plot(test_losses, 'o-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

############################
# Main Experiment Script
############################
if __name__ == "__main__":
    # Settings
    # Dataset1: Pretraining dataset
    O = 2
    T = 500
    S = 200
    sigma_pre = 0.3        # Noise for pretraining set
    sigma_classify = 0.5   # Noise for classification set

    F_list_pre = [
        # Piano-like variants (around 210Hz base)
        [200, 400, 600, 800, 1000, 1200],      # Shifted down from 210
        [215, 430, 645, 860, 1075, 1290],      # Shifted up from 210
        [205, 410, 615, 820, 1025, 1230],      # Between 200-210
        
        # Guitar-like variants (around 195Hz base)  
        [190, 380, 570, 760, 950, 1140],       # Shifted down from 195
        [200, 400, 600, 800, 1000, 1200],      # Shifted up from 195
        [192, 384, 576, 768, 960, 1152],       # Slight variation
        
        # Mixed harmonic structures (preparing for combined classes)
        [190, 205, 380, 410, 570, 820],        # Mixed fundamentals (190+205)
        [200, 215, 400, 430, 600, 860],        # Mixed fundamentals (200+215)
        [192, 210, 384, 420, 630, 840],        # Close to actual combined
        
        # Additional variations for robustness
        [198, 396, 594, 792, 990, 1188],       # In-between frequency
        [207, 414, 621, 828, 1035, 1242],      # Another in-between
        [193, 386, 579, 772, 965, 1158],       # Slight guitar variant
    ]
    P_list_pre = [
        # Piano-like power distributions (strong harmonics)
        np.array([[14, 11, 9, 7, 5, 3],
                [8, 6, 5, 3, 2, 1]]),
        
        np.array([[13, 10, 8, 6, 4, 2],
                [7, 5, 4, 3, 2, 1]]),
        
        np.array([[15, 12, 10, 8, 5, 3],
                [9, 7, 6, 4, 3, 2]]),
        
        # Guitar-like power distributions (strong fundamental, rapid decay)
        np.array([[20, 8, 6, 5, 3, 2],
                [12, 5, 3, 2, 1, 0.5]]),
        
        np.array([[18, 7, 5, 4, 3, 2],
                [10, 4, 3, 2, 1, 0.5]]),
        
        np.array([[22, 9, 7, 6, 4, 2],
                [13, 6, 4, 3, 2, 1]]),
        
        # Mixed power distributions
        np.array([[10, 10, 8, 8, 6, 4],
                [7, 7, 5, 5, 3, 2]]),
        
        np.array([[8, 12, 6, 10, 5, 4],
                [6, 8, 4, 6, 3, 2]]),
        
        np.array([[9, 9, 7, 7, 5, 3],
                [6, 6, 4, 4, 3, 2]]),
        
        # Additional varied distributions
        np.array([[16, 10, 8, 6, 4, 2],
                [9, 6, 5, 4, 2, 1]]),
        
        np.array([[11, 9, 7, 5, 3, 2],
                [7, 5, 4, 3, 2, 1]]),
        
        np.array([[19, 8, 6, 5, 3, 2],
                [11, 5, 4, 3, 2, 1]]),
    ]
    # Dataset2: Classification dataset (DIFFERENT F and P)
    F_list_class = [
        [210, 420, 630, 840, 1050, 1260],     # Class A: Piano (harmonics of 210Hz)
        [195, 390, 585, 780, 975, 1170],      # Class B: Guitar (harmonics of 195Hz)
        [195, 210, 390, 420, 630, 840],       # Class C: Combined, Guitar-dominant
        [195, 210, 420, 630, 840, 1050],      # Class D: Combined, Piano-dominant  
        [195, 210, 390, 585, 780, 975],       # Class E: Combined, Balanced
    ]
    P_list_class = [
        # Class A: Piano solo - rich harmonic content as mentioned in manuscript
        np.array([[15, 12, 10, 8, 4, 3],    # Dim 0: main response
                [9, 7, 6, 4, 3, 2]]),     # Dim 1: could represent different electrode

        # Class B: Guitar solo - strong F0, weaker harmonics  
        np.array([[12, 5, 5, 4, 4, 3],  # Very strong fundamental
                [11, 4, 2, 1, 0.5, 0.3]]),
        
        # Class C: Guitar-primed combined (enhanced guitar harmonics)
        np.array([[15, 8, 10, 7, 5, 3],    # Enhanced 195Hz and its harmonics
                [10, 7, 5, 6, 4, 2]]),
        
        # Class D: Piano-primed combined (enhanced piano harmonics)  
        np.array([[6, 12, 10, 10, 10, 5],    # Enhanced 210Hz and its harmonics
                [5, 9, 7, 6, 5, 4]]),
        
        # Class E: No priming/balanced combined
        np.array([[7, 7, 4, 4, 3, 3],     # Equal representation
                [6, 6, 3, 3, 2, 2]]),
    ]
    device = 'cpu'
    hidden_dim = 128

    # User toggles:
    pretrain_with_noise = True
    classify_with_noise = True
    do_pretrain = False   # Toggle to compare w/o pretraining


    # 1. Pretraining
    print('\n###################\n   PRETRAINING\n###################')
    # Build pretraining dataset
    sigma = sigma_pre if pretrain_with_noise else 0.0
    pretrain_loader, pretest_loader, input_dim, num_classes = build_dataset(
        O, F_list_pre, P_list_pre, T, S, sigma, split=(0.85, 0.15), seed=0)

    # Initialize model
    rnn = RNNClassifierReconstructor(input_dim, hidden_dim, num_classes).to(device)

    if do_pretrain:
        pretrain_epochs = 200
        print("Pretraining on reconstruction loss...")
        train_losses_pt, test_losses_pt = pretrain_reconstruction(
            rnn, pretrain_loader, pretest_loader, epochs=pretrain_epochs, reconstruction_time=0, device=device)
        plot_losses(train_losses_pt, test_losses_pt, title="Pretraining: Reconstruction loss")
        # Save a copy of weights after pretraining
        pretrained_state_dict = copy.deepcopy(rnn.state_dict())
    else:
        pretrained_state_dict = None  # not used

    # 2. Classification with different dataset
    print('\n###################\n  CLASSIFICATION\n###################')
    sigma = sigma_classify if classify_with_noise else 0.0
    train_loader, test_loader, input_dim, num_classes = build_dataset(
        O, F_list_class, P_list_class, T, S, sigma, split=(0.85, 0.15), seed=101)

    ###### With/without pretraining toggle
    # "reset" the model for classification task (i.e. new instance for comparison)
    rnn_clf = RNNClassifierReconstructor(input_dim, hidden_dim, num_classes).to(device)
    if do_pretrain:
        print('Loading pretrained RNN weights for classification...')
        rnn_clf.load_state_dict(pretrained_state_dict)
    else:
        print('Training classification from scratch (no pretraining)...')

    # 3. Train classification head
    clf_epochs = 200
    train_accs, test_accs, train_losses, test_losses = train_classification(
        rnn_clf, train_loader, test_loader, epochs=clf_epochs, device=device)

    # 4. Plot classification curves
    title = "Classification (with Pretraining)" if do_pretrain else "Classification (no Pretraining)"
    plot_classification(train_accs, test_accs, train_losses, test_losses, title=title)