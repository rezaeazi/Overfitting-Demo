import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Synthetic data generation
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
    
model = SimpleNet()

# loss & optimizer
critrion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Model training
epochs = 100
train_losses = []

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = critrion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
# Draw overfitting graph (train loss only)
plt.plot(train_losses, label="train loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Demo")
plt.legend()
plt.show()

# Save model
torch.save(model.state_dict(), "../model/simple_net.pth")


    
    
        