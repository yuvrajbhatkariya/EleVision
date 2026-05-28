"""
train_elephant.py
Train a binary classifier (Elephant / Other) from WAV files using ResNet18 on mel-spectrograms.
"""

import os
import random
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# Config
# -------------------------
DATA_DIR = "voice"             # root folder (elephant/, other/)
SAMPLE_RATE = 16000
DURATION = 5                   # seconds
SAMPLES = SAMPLE_RATE * DURATION

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5

VAL_SPLIT = 0.2
NUM_WORKERS = 2

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "best_model.pth"

AUGMENT_PROB = 0.5

# -------------------------
# Seed Everything
# -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# -------------------------
# Dataset
# -------------------------
class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=SAMPLE_RATE, duration=DURATION, n_mels=N_MELS,
                 n_fft=N_FFT, hop_length=HOP_LENGTH, augment=False):
        self.root_dir = root_dir
        self.sr = sr
        self.samples = sr * duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment

        self.files = []
        self.labels = []

        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            wavs = glob(os.path.join(root_dir, c, "*.wav"))
            for w in wavs:
                self.files.append(w)
                self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.files)

    def load_audio(self, path):
        y, sr = librosa.load(path, sr=self.sr)

        # ensure fixed 5-second length
        if len(y) < self.samples:
            y = np.pad(y, (0, self.samples - len(y)), mode='constant')
        else:
            y = y[:self.samples]

        return y

    def add_noise(self, y, noise_factor=0.003):
        return y + noise_factor * np.random.randn(len(y))

    def time_shift(self, y, shift_max=0.2):
        shift = int(random.uniform(-shift_max, shift_max) * len(y))
        return np.roll(y, shift)

    def compute_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
        return mel_norm.astype(np.float32)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        y = self.load_audio(path)

        # Augment only in training
        if self.augment and random.random() < AUGMENT_PROB:
            if random.random() < 0.5:
                y = self.add_noise(y, noise_factor=random.uniform(0.001, 0.01))
            else:
                y = self.time_shift(y)

        mel = self.compute_mel(y)
        mel = np.expand_dims(mel, axis=0)  # (1, mel_bins, time)

        return torch.tensor(mel), torch.tensor(label, dtype=torch.long)


# -------------------------
# Model: Updated for PyTorch 2.x
# -------------------------
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18(num_classes=2):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # modify first conv layer to accept 1 channel
    w = model.conv1.weight.data
    new_conv = nn.Conv2d(
        1, w.shape[0], kernel_size=7, stride=2, padding=3, bias=False
    )
    new_conv.weight.data = w.mean(dim=1, keepdim=True)
    model.conv1 = new_conv

    # replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    losses, preds, trues = [], [], []

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(out.argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    return np.mean(losses), accuracy_score(trues, preds)


def evaluate(model, loader, criterion):
    model.eval()
    losses, preds, trues = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            loss = criterion(out, y)

            losses.append(loss.item())
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())

    return np.mean(losses), accuracy_score(trues, preds), trues, preds


# -------------------------
# Main
# -------------------------
def main():
    dataset = AudioDataset(DATA_DIR)
    n_total = len(dataset)
    n_val = int(VAL_SPLIT * n_total)
    n_train = n_total - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_set = torch.utils.data.Subset(AudioDataset(DATA_DIR, augment=True), train_set.indices)
    val_set = torch.utils.data.Subset(AudioDataset(DATA_DIR, augment=False), val_set.indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = get_resnet18(num_classes=len(dataset.class_to_idx)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            torch.save({
                "model": model.state_dict(),
                "class_to_idx": dataset.class_to_idx,
                "epoch": epoch
            }, CHECKPOINT_PATH)

            best_acc = val_acc
            print(f"🔥 Saved new best model (Acc {val_acc*100:.2f}%)!")

    print("\nFinal Evaluation:")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(
        y_true, y_pred, target_names=list(dataset.class_to_idx.keys())
    ))


if __name__ == "__main__":
    main()
