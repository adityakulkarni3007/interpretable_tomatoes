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
# Add a logger
import wandb
import hydra
import omegaconf
# Multi-class classification metrics
from sklearn.metrics import top_k_accuracy_score
from datetime import datetime
# Set directory
os.chdir('/home/aditya/Code/interpretable_ml/project')

@hydra.main(config_path='config', config_name='config')
def train(cfg):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # Set up wandb
    if cfg.model == 'resnet':
        model = ResNet(cfg.num_classes).to(device)
    elif cfg.model == 'custom_cnn':
        model = CNN_NeuralNet(3, cfg.num_classes).to(device)
    wandb.init(project='tomato-disease-classification', name=f'{cfg.model}_{datetime.now()}',\
               job_type='train', config=omegaconf.OmegaConf.to_container(cfg))
    # model = CNN_NeuralNet(3, 5).to(device)
    # Define the loss function
    criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Define the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # Set seed
    torch.manual_seed(cfg.seed)
    if cfg.dataset == 'CCMT':
        dataset = TomatoDataset(f'{cfg.root_dir}/archive/CCMT_FInal Dataset', transform)
    elif cfg.dataset == 'PlantVillage':
        dataset = TomatoDataset(f'{cfg.root_dir}/plantdataset/plantvillage dataset/color', transform)

    if os.path.exists(f'{cfg.root_dir}/stratified_train_indices.txt') and \
        os.path.exists(f'{cfg.root_dir}/stratified_val_indices.txt') and \
        os.path.exists(f'{cfg.root_dir}/stratified_test_indices.txt'):
        with open(f'{cfg.root_dir}/stratified_train_indices.txt', 'r') as f:
            stratified_train_indices = [int(line.rstrip('\n')) for line in f]
        with open(f'{cfg.root_dir}/stratified_val_indices.txt', 'r') as f:
            stratified_val_indices = [int(line.rstrip('\n')) for line in f]
        with open(f'{cfg.root_dir}/stratified_test_indices.txt', 'r') as f:
            stratified_test_indices = [int(line.rstrip('\n')) for line in f]
        print(f"Loaded pre-computed stratified splits")
    else:
        print(f"Computing stratified splits")
        # Make a stratified split 
        stratified_train_indices = []
        stratified_val_indices = []
        stratified_test_indices = []
        class_indices = {}
        for i in range(len(dataset)):
            if dataset[i][1] not in class_indices:
                class_indices[dataset[i][1]] = []
            class_indices[dataset[i][1]].append(i)
        for _, class_indices in class_indices.items():
            # Set seed
            np.random.seed(cfg.seed)
            np.random.shuffle(class_indices)
            train_indices = class_indices[:int(0.7*len(class_indices))]
            val_indices = class_indices[int(0.7*len(class_indices)):int(0.8*len(class_indices))]
            test_indices = class_indices[int(0.8*len(class_indices)):]
            stratified_train_indices.extend(train_indices)
            stratified_val_indices.extend(val_indices)
            stratified_test_indices.extend(test_indices)
        # Save the indices
        # np.save('stratified_train_indices.npy', stratified_train_indices)
        # np.save('stratified_val_indices.npy', stratified_val_indices)
        # np.save('stratified_test_indices.npy', stratified_test_indices)
        # Save indices as .txt files
        with open(os.path.join(cfg.root_dir, "stratified_train_indices.txt"), 'w') as f:
            for s in stratified_train_indices:
                f.write(str(s) + '\n')
        with open(os.path.join(cfg.root_dir, "stratified_val_indices.txt"), 'w') as f:
            for s in stratified_val_indices:
                f.write(str(s) + '\n')
        with open(os.path.join(cfg.root_dir, "stratified_test_indices.txt"), 'w') as f:
            for s in stratified_test_indices:
                f.write(str(s) + '\n')
    
    # Add splits and indices as artefacts to wandb
    artifact = wandb.Artifact('stratified_split', type='dataset')
    artifact.add_file(f'{cfg.root_dir}/stratified_train_indices.txt')
    artifact.add_file(f'{cfg.root_dir}/stratified_val_indices.txt')
    artifact.add_file(f'{cfg.root_dir}/stratified_test_indices.txt')
    wandb.log_artifact(artifact)

    train_dataset = torch.utils.data.Subset(dataset, stratified_train_indices)
    val_dataset = torch.utils.data.Subset(dataset, stratified_val_indices)
    test_dataset = torch.utils.data.Subset(dataset, stratified_test_indices)
    del stratified_train_indices, stratified_val_indices, stratified_test_indices
    # Define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    # Train the model
    for epoch in range(cfg.num_epochs):
        # Setup pbar
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        val_step_threshold = int(len(train_dataloader) * cfg.val_interval)
        display_step_threshold = int(len(train_dataloader) * cfg.display_loss_interval)
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
            if (i+1) % display_step_threshold == 0:
                pbar.set_description(f'Epoch: {epoch+1}/{cfg.num_epochs}, Step: {i+1}/{len(train_dataloader)}, Loss: {loss.item()}')
                wandb.log({'train/loss': loss.item()})
            # Evaluate the model
            if (i+1) % val_step_threshold == 0:
                print('Evaluating the model')
                model.eval()
                with torch.no_grad():
                    gt, preds = [], []
                    for images, labels in val_dataloader:
                        images = images.to(device)
                        labels = torch.tensor([dataset.classes.index(label) for label in labels]).to(device)
                        outputs = model(images)
                        val_loss = criterion(outputs, labels)
                        wandb.log({'val/loss': val_loss.item()})
                        predictions = torch.softmax(outputs, dim=1)
                        gt.extend(labels.cpu().numpy())
                        preds.extend(predictions.cpu().numpy())
                    gt = np.array(gt)
                    preds = np.array(preds)
                    top_1_accuracy = top_k_accuracy_score(gt, preds, k=1)
                    top_3_accuracy = top_k_accuracy_score(gt, preds, k=3)
                    print(f'Top-1 Accuracy: {top_1_accuracy}, Top-3 Accuracy: {top_3_accuracy}')
                    wandb.log({'val/Top-1 Accuracy': top_1_accuracy, 'val/Top-3 Accuracy': top_3_accuracy})
                model.train()

    # Save the model
    torch.save(model.state_dict(), f'{cfg.root_dir}/model_{cfg.model}.pth')
    # Log the model
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f'{cfg.root_dir}/model_{cfg.model}.pth')
    wandb.log_artifact(artifact)

    # Test performance
    model.eval()
    gt, preds = [], []
    top_1_accuracy, top_3_accuracy = 0, 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = torch.tensor([dataset.classes.index(label) for label in labels]).to(device)
            outputs = model(images)
            predictions = torch.softmax(outputs, dim=1)
            gt.extend(labels.cpu().numpy())
            preds.extend(predictions.cpu().numpy())
        gt = np.array(gt)
        preds = np.array(preds)
        top_1_accuracy += top_k_accuracy_score(gt, preds, k=1)
        top_3_accuracy += top_k_accuracy_score(gt, preds, k=3)
    columns = ['Top-1 Accuracy', 'Top-3 Accuracy']
    values = [[top_1_accuracy/len(test_dataloader), top_3_accuracy/len(test_dataloader)]]
    wandb.Table(data=values, columns=columns)

if __name__ == '__main__':
    # Initialize sweep by passing in config.
    # sweep_cfg = omegaconf.OmegaConf.load('/home/aditya/Code/interpretable_ml/project/config/sweep_config.yaml')
    # sweep_cfg = omegaconf.OmegaConf.to_container(sweep_cfg)
    # sweep_id = wandb.sweep(sweep=sweep_cfg, project="tomato-disease-classification")

    # Start sweep job.
    # wandb.agent(sweep_id, function=train)
    train()