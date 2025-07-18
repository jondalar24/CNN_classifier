import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 1. Configurar transformaciones y dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='data/train', transform=transform),
    batch_size=4, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(root='data/val', transform=transform),
    batch_size=4, shuffle=False)

# 2. Definir el modelo CNN
class CNNBinaryClassifier(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # -> (B, 32, 64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)              # -> (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # -> (B, 64, 32, 32)
        self.pool2 = nn.MaxPool2d(2, 2)              # -> (B, 64, 16, 16)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 clases

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNNBinaryClassifier()

# 3. Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Entrenamiento del modelo
num_epochs = 5
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Evaluación en validación
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 5. Visualización de curvas de pérdida
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.show()

# 6. Mostrar algunas predicciones
def imshow(img, ax):
    img = img / 2 + 0.5
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.axis('off')

model.eval()
data_iter = iter(val_loader)
images, labels = next(data_iter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

fig, axs = plt.subplots(2, 10 * 2, figsize=(20, 2))
for idx in range(len(images)):
    row = idx // 10
    col = (idx % 10) * 2
    imshow(images[idx], axs[row, col])
    axs[row, col+1].text(0.5, 0.5, f"Actual: {labels[idx].item()}\nPredicted: {predicted[idx].item()}",
                         horizontalalignment='center', verticalalignment='center')
    axs[row, col+1].axis('off')
plt.tight_layout()
plt.show()

# 7. Calcular precisión en validación
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
