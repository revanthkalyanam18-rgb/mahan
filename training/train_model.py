import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn, optim

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("../cell_images", transform=transform)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features,2)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):

    running_loss = 0

    for images, labels in loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Epoch:", epoch+1, "Loss:", running_loss)

torch.save(model.state_dict(),"malaria_model.pth")

print("Model saved!")
