import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from dataset import TomatoDataset
from model import ResNet, CNN_NeuralNet
from tqdm import tqdm

def train():
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Define the model
    # model = ResNet(38).to(device)
    # model = ResNet(5).to(device)
    model = CNN_NeuralNet(3, 5).to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Define the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # Set seed
    torch.manual_seed(0)
    dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/archive/CCMT_FInal Dataset', transform)
    # dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/plantdataset/plantvillage dataset/color', transform)
    # Split the dataset into train, val and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    # Define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Train the model
    for epoch in range(20):
        # Setup pbar
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (images, labels) in pbar:
            images = images.to(device)
            labels = torch.tensor([dataset.classes.index(label) for label in labels]).to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print the loss
            if (i+1) % 10 == 0:
                pbar.set_description(f'Epoch: {epoch+1}/{20}, Step: {i+1}/{len(train_dataloader)}, Loss: {loss.item()}')
            # Evaluate the model
            if (i+1) % 100 == 0:
                print('Evaluating the model')
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in val_dataloader:
                        images = images.to(device)
                        labels = torch.tensor([dataset.classes.index(label) for label in labels]).to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print(f'Accuracy: {100*correct/total}')
                model.train()
    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    # Test performance
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = torch.tensor([dataset.classes.index(label) for label in labels]).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Test Accuracy: {100*correct/total}')

if __name__ == '__main__':
    train()