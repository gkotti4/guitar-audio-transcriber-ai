from enum import Enum
import numpy as np

import librosa
import librosa.display

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from dataset_loader import AudioDatasetLoader, validate_dataset_name, AVAILABLE_DATASETS
from onset_segmenter import OnsetSegmenter

from math_utils import relu, relu_derivative, softmax
from display_utils import plot_data
from datatypes import InitMethod



# region KNN
def KNN():
    # K Nearest-Neighbors,
    # === Step 1: Load Dataset ===
    loader = AudioDatasetLoader()
    X, y, label_encoder = loader.load(dataset_name="Kaggle_Electric_Open_Notes")

    # === Step 2: Normalize MFCCs ===
    scaler = StandardScaler() # Transforms data to have - Mean = 0, Standard Deviation = 1
    X_scaled = scaler.fit_transform(X) # Standard Scalar uses the basic formula X_scaled = (X - mean) / (std_dev)

    # === Step 3: Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    ) # X_train - input feature vectors (MFCC coefficient - each row is MFCC values for a note), 
        # y_train - correct output/labels
    
    # === Step 4: Train KNN Classifier ===
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)

    # === Step 5: Predict and Evaluate ===
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # === Step 6: Confusion Matrix ===
    note_labels = label_encoder.classes_

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=note_labels,
                yticklabels=note_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# endregion


# region Logistic Regression
'''
Multiclass Logistic Regression - Flow chart:
        Input Features (X)
        |
        v
    Compute Z = XW + b    <-- (Linear combination, shape: [n_samples x n_classes])
        |
        v
    Apply Softmax on Z   <-- (Convert logits to probabilities)
        |
        v
    Get Predicted Probabilities (Ŷ)  <-- [n_samples x n_classes]
        |
        v
    Compare to True Labels (Y)  <-- (One-hot encoded or class indices)
        |
        v
    Compute Cross-Entropy Loss
        |
        v
    Backpropagate Error (Compute Gradients: dW, db)
        |
        v
    Update Weights and Bias using Gradient Descent
        |
        v
        Repeat for next epoch
'''

class LogisticRegressionClassifier:
    def __init__(self, lr=0.0005):
        '''
            n_features - number of input features per sample (width (number of columns) for X input matrix) - input dim

        '''
        self.lr=lr
        self.W = None 
        self.b = None 
        self.losses = []

        
    def to_one_hot(self, y, n_classes):
        return np.eye(n_classes)[y]

    def compute_Z(self, X):
        '''
            Formula: Z = XW + b     (shape: [n_samples x n_classes])
            Purpose: compute logits (Z), raw outputs of the model before activation
        '''
        Z = np.dot(X, self.W) + self.b     # produces shape: [n_samples, n_classes] (X shape: [n_samples, n_features] @ W shape: [n_features, n_classes])
        return Z

    def softmax(self, Z):
        '''
            Purpose: apply softmax to raw outputs of Z (shape: n_samples x n_classes)
            Return: normalized probabilities per class 
        '''
        # assume Z is np.array
        # np.exp(x) = e^(x)
        Z = Z - np.max(Z) # numerical stability - if any Z_i is large, e^(Z_i) becomes astronomically huge
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_output = e_Z / np.sum(e_Z, axis=1, keepdims=True) # axis=1 operate on columns
        return softmax_output
        

    def forward(self, X):
        '''
            Purpose: Compute logits (Z) and apply softmax
            Return: Y_hat probabilities (softmax output)

            Input:      X
            Parameters: W and b
            Output:     Y_hat (n_samples x n_classes)
        '''
        Z = self.compute_Z(X)
        Y_hat = self.softmax(Z)
        return Y_hat

    def predict_class_indices(self, Y_hat):
        '''
            Purpose: Convert softmax probabilities into predicted class indices.
            
            Input:
                Y_hat: ndarray of shape (n_samples, n_classes)
                    Each row contains predicted probabilities for each class.
            
            Returns:
                predictions: ndarray of shape (n_samples,)
                            Each entry is the index of the class with the highest probability.
        '''            
        predictions = np.argmax(Y_hat, axis=1) # axis=1, operate horizontally across rows
        return predictions 

    def compute_loss(self, X, Y_hat, Y_true):
        '''
            Cross-entropy loss for multiclass classification.

            Inputs:
                Y_hat  : predicted probabilities from softmax (shape: [n_samples, n_classes])
                Y_true : one-hot encoded true labels          (shape: [n_samples, n_classes])

            Output:
                avg_loss : scalar value representing the average cross-entropy loss across all samples
        '''
        n_samples = X.shape[0] # ex. (10, 13) shape we would have 10 samples with 13 features
        
        # Step 1: Add epsilon to prevent numerical instability.
        # Why? log(0) is undefined (produces -inf), so we clip values to avoid this.
        # Step 2: Clamp predicted probabilities to be within [epsilon, 1 - epsilon].
        # This ensures all log operations are safe and prevents exploding gradients during training.
        epsilon = 1e-12
        Y_hat_clipped = np.clip(Y_hat, epsilon, 1 - epsilon)

        # Step 3: Calculate individual cross-entropy loss per sample.
        # For each row (sample), only the log probability of the correct class is kept:
        # - `Y_true` is one-hot, so only the correct class entry contributes (1 * log(p))
        # - We sum along axis=1 to collapse per-class values into a single scalar per sample.
        individual_losses = -np.sum(Y_true * np.log(Y_hat_clipped), axis=1)

        # Step 4: Compute the average of those individual losses across all samples in the batch.
        # This scalar loss is what we use to guide gradient descent.
        avg_loss = np.mean(individual_losses) # 1/N * individual_losses

        # Step 5: Return the final average cross-entropy loss.
        return avg_loss

    def backward(self, X, Y_hat, Y_true):
        '''
            Purpose: compute gradients for weights/bias
            Inputs:
                X: input features (n_samples x n_features)
                Y_hat: predicted softmax outputs (n_samples x n_classes)
                Y_true: one-hot true labels (n_samples x n_classes)
            
            Returns:
                dW: gradient of weights (n_features x n_classes)
                db: gradient of biases (n_classes,)
        '''
        n_samples = X.shape[0]

        # Derivative of loss with respect to logits Z
        dZ = Y_hat - Y_true # shape: (n_samples x n_classes)

        # Gradient with respect to weights
        dW = (1 / n_samples) * np.dot(X.T, dZ) # shape: (n_features x n_classes)

        # Gradient with respect to biases
        db = (1 / n_samples) * np.sum(dZ, axis=0) # shape: (n_classes,)

        return dW, db
    
    def update_parameters(self, W_gradient, b_gradient):
        # last step of gradient descent
        self.W -= self.lr * W_gradient
        self.b -= self.lr * b_gradient


    def train(self, epochs=20000): #(self, X, Y_true, epochs=10):
        # region Epoch:
        #   forward
        #   compute loss
        #   backward
        #   update parameters
        #   (return) predicted class indicies from forward
        # endregion

        # === Load Dataset ===
        loader = AudioDatasetLoader()
        X_train, X_test, y_train, y_test = loader.train_test_split()
        
        # === Auto-Detect Dimensions ===
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train))

        # === One-Hot Encode Y ===
        y_train_one_hot = self.to_one_hot(y_train, n_classes) # We must do this to allow it class indices to work with the matrices. Ex. class index 2 -> [0,0,1,0,0,0]


        # === Init Weights & Biases ===
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)

        # === Train Loop ===
        for i in range(epochs):
            Y_hat = self.forward(X_train)
            loss = self.compute_loss(X_train, Y_hat, y_train_one_hot)
            dW, db = self.backward(X_train, Y_hat, y_train_one_hot)
            self.update_parameters(dW, db)

            # === Logging & Metrics ===
            self.losses.append(loss)
            if i % 100 == 0:
                predictions = self.predict_class_indices(Y_hat)
                acc = np.mean(predictions == y_train)
                print(f"Epoch {i+1}/{epochs} - Loss: {loss:.4f} - Train Acc: {acc:.2f}")
        
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

def LogReg(): # logistic regression (multiclass - softmax)
    logreg = LogisticRegressionClassifier()
    logreg.train()

# endregion


# region SVM
def SVM():
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix
    from dataset_loader import AudioDatasetLoader

    # === Load Dataset ===
    loader = AudioDatasetLoader()
    X, y, label_encoder = loader.load()  # already returns features and encoded labels

    # === Train/Test Split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Create and Train Linear SVM ===
    model = LinearSVC(C=1.0, max_iter=10000)
    model.fit(X_train, y_train)

    # === Predict ===
    y_pred = model.predict(X_test)

    # === Evaluate ===
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# endregion



# region MLP

# Input (X) -> [Linear -> Activation] -> ... [Linear -> Softmax] -> Output
class MLPClassifier():
    def __init__(self, hidden_dim, lr=0.00025, dataset_name=None):
        self.hidden_dim=hidden_dim
        self.dname=validate_dataset_name(dataset_name)
        self.lr=lr
        self.W1=None
        self.b1=None
        self.W2=None
        self.b2=None
        self.losses=[]

    def forward(self, X):
        # Z1 = X*W1 + b1
        # A1 = ReLU(Z1)
        # Z2 = A1*W2 + b2
        # Yhat = softmax(Z2)
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.Y_hat = softmax(self.Z2)
        return self.Y_hat

    def cross_entropy(self, Y_hat, Y_true):
        eps = 1e-12
        Y_hat = np.clip(Y_hat, eps, 1 - eps)
        return -np.mean(np.sum(Y_true * np.log(Y_hat), axis=1))
    
    def backward(self, X, Y_hat, Y_true):
        m = X.shape[0]

        # Step 1: gradient of loss wrt Z2
        dZ2 = Y_hat - Y_true                     # (m × output_dim)

        # Step 2: gradients for W2 and b2
        dW2 = np.dot(self.A1.T, dZ2) / m          # (hidden_dim × output_dim)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Step 3: backprop into hidden layer
        dA1 = np.dot(dZ2, self.W2.T)             # (m × hidden_dim)
        dZ1 = dA1 * relu_derivative(self.Z1)     # apply ReLU’s derivative

        # Step 4: gradients for W1 and b1
        dW1 = np.dot(X.T, dZ1) / m               # (input_dim × hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # regularization
        #reg_lambda = 1e-6
        #dW1 += reg_lambda * self.W1
        #dW2 += reg_lambda * self.W2

        # Step 5: gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, X_test=None, y_test=None, epochs=10000, log_every=250, init_method=InitMethod.Xavier):
        #np.random.seed(42)

        n_samples, input_dim = X.shape
        output_dim = len(np.unique(y))

        # weights initialization
        if init_method == InitMethod.Xavier:
            limit1 = np.sqrt(6/(input_dim + self.hidden_dim))
            self.W1 = np.random.uniform(-limit1, limit1, (input_dim, self.hidden_dim))

            limit2 = np.sqrt(6/(self.hidden_dim + output_dim))
            self.W2 = np.random.uniform(-limit2, limit2, (self.hidden_dim, output_dim))

        elif init_method == InitMethod.HE:
            def he(fan_in, fan_out):
                std = np.sqrt(2.0 / fan_in)
                return np.random.randn(fan_in, fan_out) * std
            self.W1 = he(input_dim, self.hidden_dim)
            self.W2 = he(self.hidden_dim, output_dim)

        else:
            self.W1 = np.random.randn(input_dim, self.hidden_dim) * 0.1
            self.W2 = np.random.randn(self.hidden_dim, output_dim) * 0.1

        self.b1 = np.zeros((1, self.hidden_dim))
        self.b2 = np.zeros((1, output_dim))

        # one‐hot encode y
        Y_one_hot = np.eye(output_dim)[y]

        for epoch in range(1, epochs+1):
            # forward / backward / update
            Y_hat = self.forward(X)
            loss  = self.cross_entropy(Y_hat, Y_one_hot)
            self.backward(X, Y_hat, Y_one_hot)
            self.losses.append(loss)

            # logging
            if epoch % log_every == 0 or epoch == 1:
                train_acc = (np.argmax(Y_hat, axis=1) == y).mean()
                msg = f"Epoch {epoch}/{epochs} — loss: {loss:.4f} — train_acc: {train_acc:.2f}"
                if X_test is not None:
                    val_hat = self.forward(X_test)
                    val_acc = (np.argmax(val_hat, axis=1) == y_test).mean()
                    msg += f" — val_acc: {val_acc:.2f}"
                print(msg)

    def train(self, epochs=25000):
        loader = AudioDatasetLoader(dataset_name=self.dname, test_size=.1)
        X_train, X_test, y_train, y_test = loader.train_test_split()

        self.fit(X_train, y_train, X_test, y_test, epochs)

        print(f"Final Test Accuracy: {((np.argmax(self.forward(X_test),axis=1)==y_test).mean()):.2f}")
        plot_data(self.losses, "Epoch", "Loss", "Cross Entropy Loss",)



def MLP(hidden_dim=64, dataset=0):
    net = MLPClassifier(hidden_dim, AVAILABLE_DATASETS[dataset])
    net.train()

# endregion



# region CNN
# region CNN Description
"""
CNNClassifier forward pass flow:

Input: 
    x (Tensor) of shape (batch_size, 1, n_mels, time_frames)
    - batch_size: number of audio clips in the batch
    - 1: single channel (mel-spectrogram)
    - n_mels: frequency bins (e.g. 64)
    - time_frames: number of time steps per clip

Layers:

1. Conv2d(1 → 16, kernel_size=3, padding=1)
    • Each of the 16 filters slides over the mel-spectrogram,
      computing local weighted sums over 3×3 patches.
    • Output shape: (batch_size, 16, n_mels, time_frames)

2. ReLU(inplace=True)
    • Applies element-wise max(0,·), zeroing out negative activations.
    • Shape unchanged: (batch_size, 16, n_mels, time_frames)

3. MaxPool2d(kernel_size=2, stride=2)
    • Downsamples by taking the maximum in each non-overlapping 2×2 block.
    • Cuts both frequency and time dimensions in half.
    • Output shape: (batch_size, 16, n_mels/2, time_frames/2)

4. Conv2d(16 → 32, kernel_size=3, padding=1)
    • Learns 32 new feature maps over the pooled inputs.
    • Output shape: (batch_size, 32, n_mels/2, time_frames/2)

5. ReLU(inplace=True)
    • Zeroes out negative values in the 32 feature maps.
    • Shape unchanged: (batch_size, 32, n_mels/2, time_frames/2)

6. MaxPool2d(kernel_size=2, stride=2)
    • Further downsamples by 2×2 pooling.
    • Output shape: (batch_size, 32, n_mels/4, time_frames/4)

7. AdaptiveAvgPool2d((1,1))
    • Globally averages each of the 32 feature maps to a single value.
    • Collapse spatial dimensions to 1×1.
    • Output shape: (batch_size, 32, 1, 1)

8. Flatten()
    • Removes the final singleton dimensions.
    • Output shape: (batch_size, 32)

9. Linear(32 → num_classes)
    • Fully connected to one logit per class.
    • Output shape: (batch_size, num_classes)

Result:
    A tensor of raw class scores (“logits”) for each input in the batch.
    During training these go into CrossEntropyLoss; during inference you
    take argmax over the class dimension to get predicted note labels.
"""

# endregion

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class MelNoteDataset(Dataset):
    """Loads mel-spectrograms on the fly from file paths + labels."""
    def __init__(self, file_paths, labels,
                n_mels=128, sr=44100, hop_length=256, duration=0.425):
        self.paths      = file_paths
        self.labels     = labels
        self.n_mels     = n_mels
        self.sr         = sr
        self.hop        = hop_length
        self.duration   = duration      # used for different size samples

        print(f"Number of samples: {len(file_paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        # 1) load audio
        y, _ = librosa.load(path, sr=self.sr, mono=True, duration=self.duration)
        
        # 2) mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop,
            fmax=self.sr // 2,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # 3) normalize to [0,1]
        S_norm = (S_db + 80) / 80

        # 4) to tensor: (C=1, H=n_mels, W=frames)
        mel = torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0)
        return mel, label
    
    def print_paths(self):
        for path in self.paths:
            print(path)


class CNNClassifier(nn.Module):
    def __init__(self, num_classes, lr=1e-3, weight_decay=0.0):
        super().__init__()
        self.lr = lr
        self.wd = weight_decay
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
        self.loss_history = []
        self.acc_history   = []

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    def fit(self, train_dl, val_dl=None, epochs=50, device=None, log_every=5):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        opt   = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        lossf = nn.CrossEntropyLoss()

        for ep in range(1, epochs+1):
            self.train()
            running_loss, correct = 0.0, 0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = self(Xb)
                loss   = lossf(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running_loss += loss.item() * Xb.size(0)
                correct      += (logits.argmax(1) == yb).sum().item()

            train_loss = running_loss / len(train_dl.dataset)
            train_acc  = correct      / len(train_dl.dataset)
            self.loss_history.append(train_loss)
            self.acc_history.append(train_acc)

            if ep % log_every == 0 or ep == 1:
                msg = f"[Epoch {ep}/{epochs}] loss: {train_loss:.3f} acc: {train_acc:.2%}"
                if val_dl:
                    val_loss, val_acc = self.evaluate(val_dl, device)
                    msg += f" — val_loss: {val_loss:.3f} val_acc: {val_acc:.2%}"
                print(msg)


    def evaluate(self, dl, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        lossf = nn.CrossEntropyLoss(reduction="sum")
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for Xb, yb in dl:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = self(Xb)
                total_loss += lossf(logits, yb).item()
                correct    += (logits.argmax(1) == yb).sum().item()
        avg_loss = total_loss / len(dl.dataset)
        acc      = correct    / len(dl.dataset)
        return avg_loss, acc


def CNN(batch_size=64, test_size=0.2, seed=42):
    # ——— Prepare data ———
    dname = AVAILABLE_DATASETS[0]
    loader = AudioDatasetLoader(dataset_name=dname)
    paths, labels, label_encoder = loader.load_files_and_labels()
    Xtr, Xte, ytr, yte = train_test_split(
        paths, labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
    train_dl = DataLoader(MelNoteDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(MelNoteDataset(Xte, yte), batch_size=batch_size)

    # ——— Train & evaluate ———
    model = CNNClassifier(num_classes=len(label_encoder.classes_))
    model.fit(train_dl, val_dl=test_dl, epochs=50, device=None, log_every=5)
    test_loss, test_acc = model.evaluate(test_dl)
    print(f"\nFinal Test — loss: {test_loss:.3f} — acc: {test_acc:.2%}")

    # ——— Plot learning curves ———
    plot_data(
        {"train_loss": model.loss_history, "train_acc": model.acc_history},
        xlabel="Epoch",
        ylabel="Value",
        title="CNN Training Curves",
    )
    return model

# endregion


if __name__ == "__main__":
    #KNN()
    #LogReg()
    #SVM()
    #MLP()
    #CNN()
    pass





























'''
        def train(self, epochs=25000, seed=42):
            loader = AudioDatasetLoader()
            X_train, X_test, y_train, y_test = loader.train_test_split()

            np.random.seed(seed)

            n_samples, input_dim = X_train.shape
            output_dim = len(np.unique(y_train))

            self.W1 = np.random.randn(input_dim, self.hidden_dim) * 0.1
            self.b1 = np.zeros((1, self.hidden_dim))
            self.W2 = np.random.randn(self.hidden_dim, output_dim) * 0.1
            self.b2 = np.zeros((1, output_dim))
        
            y_one_hot = np.eye(self.b2.shape[1])[y_train]

            # Train
            for epoch in range(epochs):
                Y_hat = self.forward(X_train)
                loss = self.cross_entropy(Y_hat, y_one_hot)
                self.backward(X_train, Y_hat, y_one_hot)

                if epoch % 250 == 0:
                    preds = np.argmax(self.Y_hat, axis=1)
                    acc = np.mean(preds == y_train)
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.2f}")

            # Test
            Y_test_hat = self.forward(X_test)
            test_preds = np.argmax(Y_test_hat, axis=1)
            test_acc = np.mean(test_preds == y_test)
            print(f"Test Accuracy: {test_acc:.2f}")
'''