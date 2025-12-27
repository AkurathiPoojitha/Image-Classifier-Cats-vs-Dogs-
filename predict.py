import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN()
model.load_state_dict(torch.load("cat_dog_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

output = model(img)
_, pred = torch.max(output, 1)

print("Prediction:", "Cat 🐱" if pred.item() == 0 else "Dog 🐶")
