import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Gerät auswählen (GPU falls vorhanden, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST-Testdaten laden
transform = transforms.ToTensor()
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
# This is the comment 
# Modellstruktur wiederherstellen (muss identisch mit dem Trainingsmodell sein)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),               # 28x28 → 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)         # 10 Klassen (Ziffern 0–9)
        )

    def forward(self, x):
        return self.model(x)

# Modellinstanz erstellen und auf das Gerät verschieben
model = MLP().to(device)

# ✅ Modellparameter aus Datei laden
# Wenn die .pt-Datei im Projekt-Hauptverzeichnis ist:
model.load_state_dict(torch.load("../mlp_ziffern_model.pt", map_location=device, weights_only=False))

# Falls sich die .pt-Datei im selben Ordner wie dieses Skript befindet, dann:
# model.load_state_dict(torch.load("mlp_ziffern_model.pt", map_location=device, weights_only=False))

model.eval()
print("✅ Modell erfolgreich geladen!")

# Ein Testbild aus dem Loader nehmen
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

# Vorhersage berechnen
with torch.no_grad():
    output = model(images)
    prediction = torch.argmax(output, dim=1)

# Bild anzeigen
plt.imshow(images[0].cpu().squeeze(), cmap="gray")
plt.title(f"Vorhergesagte Ziffer: {prediction.item()}")
plt.axis("off")
plt.show()
