# mlp_ziffern_model.pt
# Dieses Skript lädt ein trainiertes MLP-Modell zur Ziffernerkennung (MNIST),
# zeigt ein zufälliges Testbild an und gibt die Vorhersage aus.

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# Gerät auswählen (GPU, falls verfügbar, sonst CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MLP-Modell definieren (muss exakt dem Trainingsmodell entsprechen)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),           # 28x28 → 784
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)     # 10 Ziffernklassen (0–9)
        )

    def forward(self, x):
        return self.model(x)

# Modell laden
model = MLP().to(device)
model.load_state_dict(torch.load("mlp_mnist_model.pt", map_location=device))
model.eval()

# MNIST-Testdaten vorbereiten
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Zufälliges Bild aus dem Testset auswählen
image, label = random.choice(test_dataset)

# Bild anzeigen
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Tatsächliche Ziffer: {label}")
plt.axis("off")
plt.show()

# Vorhersage vorbereiten
image = image.unsqueeze(0).to(device)  # Tensorform [1, 1, 28, 28]
output = model(image)
_, predicted = torch.max(output, 1)

# Ergebnis anzeigen
print(f"🧠 Modellvorhersage: {predicted.item()}")
