import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import zipfile
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

# -----------------------
# 1. Función: cargar imágenes desde zip
# -----------------------
def load_images_from_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        images = {'anastasia': [], 'takao': []}
        for file_name in zip_ref.namelist():
            if file_name.startswith('anastasia') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['anastasia'].append(np.array(img))
            elif file_name.startswith('takao') and file_name.endswith('.jpg'):
                with zip_ref.open(file_name) as file:
                    img = Image.open(file).convert('RGB')
                    images['takao'].append(np.array(img))
    return images

# -----------------------
# 2. Cargar y preparar datos
# -----------------------
print("[INFO] Cargando imágenes desde data.zip...")
data = load_images_from_zip("data.zip")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

X = []
y = []

for label, class_name in enumerate(['anastasia', 'takao']):
    for img_array in data[class_name]:
        img = Image.fromarray(img_array)
        img_tensor = transform(img)
        X.append(img_tensor)
        y.append(label)

X = torch.stack(X)
y = torch.tensor(y)

# Separar en train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20)

print(f"[INFO] Tamaño del dataset: {len(X)} imágenes (train={len(X_train)}, val={len(X_val)})")

# -----------------------
# 3. Definir la red CNN
# -----------------------
class AnimeCNN(nn.Module):
    def __init__(self):
        super(AnimeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = AnimeCNN()

# -----------------------
# 4. Entrenamiento
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# -----------------------
# 5. Precisión final
# -----------------------
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"[INFO] Precisión de validación: {100 * correct / total:.2f}%")

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title('Training and Validation Loss')
plt.show()

# 6.Visualización de predicciones
def imshow(img, ax):
    img = img * 0.5 + 0.5  # Desnormaliza
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))  # [C,H,W] → [H,W,C]
    ax.axis('off')

# Asegura modo evaluación
model.eval()

# Muestra un lote de validación
data_iter = iter(val_loader)
images, labels = next(data_iter)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Grid de imágenes y predicciones
num_images = len(images)
num_cols = 5
num_rows = (num_images + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols * 2, figsize=(2 * num_cols * 2, 2 * num_rows))

for idx in range(num_images):
    row = idx // num_cols
    col = (idx % num_cols) * 2
    
    # Imagen
    imshow(images[idx].cpu(), axs[row, col])
    
    # Etiqueta real vs predicción
    axs[row, col + 1].text(0.5, 0.5, f"GT: {labels[idx].item()}\nPred: {predicted[idx].item()}",
                           ha='center', va='center', fontsize=12)
    axs[row, col + 1].axis('off')

# Elimina subplots vacíos
for idx in range(num_images, num_rows * num_cols):
    row = idx // num_cols
    col = (idx % num_cols) * 2
    axs[row, col].axis('off')
    axs[row, col + 1].axis('off')

plt.tight_layout()
plt.show()
