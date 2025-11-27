from dataset_loader import AudioDatasetLoader, validate_dataset_name, AVAILABLE_DATASETS

import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, lr=0.0005, dataset_name = None):
        '''
            n_features - number of input features per sample (width (number of columns) for X input matrix) - input dim

        '''
        self.lr=lr
        self.W = None 
        self.b = None 
        self.losses = []

        self.dataset_name = dataset_name

        
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

        if(self.dataset_name == None):
            loader = AudioDatasetLoader()
        else:
            loader = AudioDatasetLoader(self.dataset_name)

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