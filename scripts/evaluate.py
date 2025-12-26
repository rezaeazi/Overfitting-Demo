import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from train import SimpleNet

# Similar data
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

model = SimpleNet()
model.load_state_dict(torch.load("../model/simple_net.pth"))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    predictes = torch.argmax(outputs, dim=1)
    accuracy = (predictes == y_test).sum().item() / len(y_test)
    
print(f"Test Accuracy: {accuracy:.2f}")