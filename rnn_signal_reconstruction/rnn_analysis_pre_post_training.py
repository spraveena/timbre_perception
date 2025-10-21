import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
import random

def generate_time_series_dataset(O, F, P, T):
    t = np.arange(T)
    sig = np.zeros((O, T), dtype=np.float32)
    F = np.array(F)
    P = np.array(P)
    phases = 2 * np.pi * np.random.rand(O, len(F))
    for o in range(O):
        for k, freq in enumerate(F):
            sig[o] += np.sqrt(P[o, k]) * np.sin(2 * np.pi * freq * t / T + phases[o, k])
    return sig.T

def build_dataset(O, F_list, P_list, T, S, sigma, split=(0.8, 0.2), seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    num_classes = 5
    all_X, all_Y = [], []
    for c in range(num_classes):
        F, P = F_list[c], P_list[c]
        for s in range(S):
            sig = generate_time_series_dataset(O, F, P, T)
            sig += np.random.normal(0, sigma, sig.shape)
            all_X.append(sig)
            all_Y.append(c)
    all_X, all_Y = np.stack(all_X), np.array(all_Y)
    idx = np.arange(len(all_Y))
    np.random.shuffle(idx)
    all_X, all_Y = all_X[idx], all_Y[idx]
    n_train = int(len(all_Y)*split[0])
    X_train, Y_train = all_X[:n_train], all_Y[:n_train]
    X_test, Y_test = all_X[n_train:], all_Y[n_train:]
    X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long)
    X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.long)
    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)
    return train_loader, test_loader, X_test, Y_test

class RNNClassifierReconstructor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, nonlinearity='tanh'):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=1,
                          nonlinearity=nonlinearity, batch_first=True)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.recon_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, h_n = self.rnn(x)
        last_hidden = out[:, -1, :]
        logits = self.class_head(last_hidden)
        recon_out = self.recon_head(out)
        return logits, recon_out, out

##### PRETRAINING #####
def pretrain_reconstruct(model, train_loader, test_loader, epochs=10, device='cpu'):
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoints = {}
    N = epochs
    for ep in range(epochs):
        model.train()
        for X, Y in train_loader:
            X = X.to(device)
            optimizer.zero_grad()
            _, recon, _ = model(X)
            loss = mse(recon, X)
            loss.backward()
            optimizer.step()
        if ep == 0:
            checkpoints['pre'] = copy.deepcopy(model.state_dict())
        if ep == N//2-1:
            checkpoints['middle'] = copy.deepcopy(model.state_dict())
        if ep == N-1:
            checkpoints['after'] = copy.deepcopy(model.state_dict())
    return checkpoints

##### POST-TRAIN (CLASSIFIER) #####
def posttrain_classifier(model, train_loader, test_loader, epochs=10, device='cpu'):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoints = {}
    N = epochs
    for ep in range(epochs):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            logits, _, _ = model(X)
            loss = ce_loss(logits, Y)
            loss.backward()
            optimizer.step()
        if ep == 0:
            checkpoints['pre'] = copy.deepcopy(model.state_dict())
        if ep == N//2-1:
            checkpoints['middle'] = copy.deepcopy(model.state_dict())
        if ep == N-1:
            checkpoints['after'] = copy.deepcopy(model.state_dict())
    return checkpoints

def extract_hidden(model, X, batch_size=32, device='cpu'):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            _, _, all_out = model(batch_X)
            last_hidden = all_out[:, -1, :]
            features.append(last_hidden.cpu().numpy())
    return np.concatenate(features, axis=0)

def plot_2d(features, labels, title):
    pca = PCA(n_components=2)
    features2d = pca.fit_transform(features)
    num_classes = len(np.unique(labels))
    plt.figure(figsize=(5,5))
    for c in range(num_classes):
        idx = labels==c
        plt.scatter(features2d[idx,0], features2d[idx,1], s=20, label=f"Class {c}", alpha=0.7)
    plt.legend()
    plt.xlabel('PCA-1'); plt.ylabel('PCA-2')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Data & Model
    O, T, S = 2, 500, 20
    sigma = 0.3
    F_list = [ [2,8,16],[2,6,12],[3,9,15],[1,5,14],[4,10,18] ]
    P_list = [
        np.array([[5,1,1],[2,2,6]]), np.array([[4,3,1],[1,5,2]]),
        np.array([[7,2,2],[2,1,7]]), np.array([[3,4,2],[4,2,3]]),
        np.array([[2,2,6],[6,1,2]])
    ]
    train_loader, test_loader, X_test, Y_test = build_dataset(O, F_list, P_list, T, S, sigma)
    input_dim, hidden_dim, num_classes = O, 128, 5
    model = RNNClassifierReconstructor(input_dim, hidden_dim, num_classes).to('cpu')

    print("=== Pretraining (Reconstruction) phase ===")
    pretrain_ckpts = pretrain_reconstruct(model, train_loader, test_loader, epochs=10)
    for phase in ['pre','middle','after']:
        model.load_state_dict(pretrain_ckpts[phase])
        features = extract_hidden(model, X_test)
        plot_2d(features, Y_test.numpy(), f"PRETRAIN ({phase}) reconstruct")

    print("=== Post-training (Classification) phase ===")
    post_ckpts = posttrain_classifier(model, train_loader, test_loader, epochs=10)
    for phase in ['pre','middle','after']:
        model.load_state_dict(post_ckpts[phase])
        features = extract_hidden(model, X_test)
        plot_2d(features, Y_test.numpy(), f"POSTTRAIN ({phase}) classifier")