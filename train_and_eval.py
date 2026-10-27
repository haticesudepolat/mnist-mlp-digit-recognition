import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 1. Cihaz Seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 2. MNIST Veri Ön İşleme ve Yükleme
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST ortalama ve standart sapması
])

# MNIST Train ve Test setlerini indir
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Train setini %85 Train, %15 Validation olarak ayır (51000 train, 9000 val)
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3. MLP Modeli (CNN Yok, 2 Gizli Katman, ReLU)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Giriş: 28x28 = 784 piksel
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),  # Hidden Layer 1
            nn.ReLU(),
            nn.Linear(256, 128),       # Hidden Layer 2
            nn.ReLU(),
            nn.Linear(128, 10)         # Çıkış: 10 sınıf (0-9)
        )

    def forward(self, x):
        return self.network(x)

model = MLP().to(device)

# 4. Kayıp Fonksiyonu ve Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Eğitim ve Değerlendirme Döngüsü
epochs = 10
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

start_time = time.time()

for epoch in range(epochs):
    # Training
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_train_loss = running_loss / total
    epoch_train_acc = (correct / total) * 100
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Validation
    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    epoch_val_loss = val_running_loss / val_total
    epoch_val_acc = (val_correct / val_total) * 100
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] - "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

total_training_time = time.time() - start_time
print(f"\nToplam Eğitim Süresi: {total_training_time:.2f} saniye")

# 6. Test Seti Değerlendirmesi
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = (test_correct / test_total) * 100
print(f"Nihai Test Accuracy: {test_accuracy:.2f}%")

# 7. Grafikleri Çiz ve Kaydet
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Grafiği')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Grafiği')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
print("Grafikler 'training_metrics.png' olarak kaydedildi.")
plt.close()

# 8. Kendi El Yazısı Rakamlarını Test Etme Fonksiyonu
def predict_custom_image(image_path, model):
    img = Image.open(image_path).convert('L')  # Gri seviye
    img = ImageOps.invert(img)                 # Beyaz kâğıt/siyah yazı -> siyah arkaplan/beyaz yazı
    img = img.resize((28, 28))                 # 28x28 boyutlandırma
    
    # Normalizasyon
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor_img = custom_transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        pred = torch.argmax(output, dim=1).item()
    return pred, img

# 'my_digits' klasöründeki görselleri test et
digits_dir = './my_digits'
if os.path.exists(digits_dir):
    files = sorted([f for f in os.listdir(digits_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if files:
        plt.figure(figsize=(len(files) * 2, 3))
        for idx, filename in enumerate(files):
            filepath = os.path.join(digits_dir, filename)
            prediction, processed_img = predict_custom_image(filepath, model)
            
            plt.subplot(1, len(files), idx + 1)
            plt.imshow(processed_img, cmap='gray')
            plt.title(f"Tahmin: {prediction}\nDosya: {filename}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('custom_digits_result.png')
        print("Kendi rakamlarının sonuçları 'custom_digits_result.png' olarak kaydedildi.")
        plt.close()