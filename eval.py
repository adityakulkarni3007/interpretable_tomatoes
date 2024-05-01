import torch
import torch.nn as nn
import numpy as np
from model import ResNet
from dataset import TomatoDataset
from torchvision import transforms
from sklearn.metrics import top_k_accuracy_score
from tqdm import tqdm

def eval():
    device = 'cuda'
    model = ResNet(5).to(device)
    model.load_state_dict(torch.load('model_bestsweep.pth'))
    # model.load_state_dict(torch.load('model_resnet_stable.pth'))
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/archive/CCMT_FInal Dataset', transform)
    with open('/home/aditya/Code/interpretable_ml/project/stratified_test_indices.txt', 'r') as f:
        stratified_test_indices = [int(line.rstrip('\n')) for line in f]
    dataset = torch.utils.data.Subset(dataset, stratified_test_indices)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    gt, preds = [], []
    for i, (img, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img = img.to(device)
        labels = torch.tensor([dataset.dataset.classes.index(label) for label in labels]).to(device)
        output = model(img)
        predictions = torch.softmax(output, dim=1)
        gt.extend(labels.detach().cpu().numpy())
        preds.extend(predictions.detach().cpu().numpy())
    
    gt = np.array(gt)
    preds = np.array(preds)
    top_1_accuracy = top_k_accuracy_score(gt, preds, k=1)
    top_3_accuracy = top_k_accuracy_score(gt, preds, k=3)
    print(f'Top-1 Accuracy: {top_1_accuracy}, Top-3 Accuracy: {top_3_accuracy}')

if __name__ == '__main__':
    eval()
