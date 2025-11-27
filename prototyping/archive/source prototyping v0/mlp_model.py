from dataset_loader import AudioDatasetLoader, validate_dataset_name, AVAILABLE_DATASETS
from math_utils import relu, relu_derivative, softmax
from display_utils import plot_data
from datatypes import InitMethod
import numpy as np

# Input (X) -> [Linear -> Activation] -> ... [Linear -> Softmax] -> Output
class MLPClassifier():
    def __init__(self, hidden_dim, lr=0.00025, dataset_name=None, duration=None):
        self.dname=validate_dataset_name(dataset_name)
        self.duration=duration
        self.hidden_dim=hidden_dim
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
        loader = AudioDatasetLoader(dataset_name=self.dname, test_size=0.2, duration=self.duration)
        X_train, X_test, y_train, y_test = loader.train_test_split()

        self.fit(X_train, y_train, X_test, y_test, epochs)

        print(f"Final Test Accuracy: {((np.argmax(self.forward(X_test),axis=1)==y_test).mean()):.2f}")
        #plot_data(self.losses, "Epoch", "Loss", "Cross Entropy Loss",)



    def forward_mfccs(self, mfccs): # Not Needed - same shape so we can direclty pass to forward()
        # Used in transcribe to pass audio segments trimmed with OnsetSegmenter into forward and return notes (y)
        return self.forward(mfccs)




def MLP(hidden_dim=64, dataset=0):
    net = MLPClassifier(hidden_dim, AVAILABLE_DATASETS[dataset])
    net.train()
    