import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np
from dataset import TomatoDataset
from model import ResNet, CNN_NeuralNet
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import AblationCAM as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

def overlay_saliency_map(input_image, saliency_map):
    """
    Overlays a saliency map on the input image.
    
    Args:
        input_image (numpy.ndarray): The input image.
        saliency_map (numpy.ndarray): The saliency map.
    
    Returns:
        numpy.ndarray: The input image with the saliency map overlaid.
    """
    # Normalize the saliency map to the range [0, 1]
    input_image = np.array(transforms.ToPILImage()(input_image))
    saliency_map = saliency_map.cpu().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
    # Convert the saliency map to a heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map), cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the input image
    overlaid_image = cv2.addWeighted(input_image, 0.5, heatmap, 0.5, 0)
    
    return overlaid_image

def saliency_map():
    device = 'cpu'
    # Load the model
    # model = ResNet(5).to(device)
    model = CNN_NeuralNet(3, 5).to(device)
    # model.load_state_dict(torch.load('model_plant.pth'))
    model.load_state_dict(torch.load('model.pth'))
    model.eval().to(device)
    # Define the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/archive/CCMT_FInal Dataset', transform)
    # Define the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    img, label = dataset[10]
    label = torch.tensor([dataset.classes.index(label)]).to(device)
    img = Variable(img.unsqueeze(0), requires_grad=True).to(device)
    print(img.shape)
    # Forward pass
    output = model(img)
    class_index = torch.argmax(output, dim=1)
    # Backward pass
    loss = output[0, class_index]
    loss.backward()
    grad = img.grad.data.abs()
    # Normalize
    # grad = (grad - grad.mean()) / grad.std()
    grad = grad.abs()
    saliency, _ = torch.max(grad,dim=1)
    ## Visualize input image and saliency map
    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # Title
    ax1.set_title('GT: ' + dataset.classes[label])

    # Show input image
    ax1.imshow(transforms.ToPILImage()(img[0]))
    ax1.axis('off')

    # Overlay the saliency map over the input image
    overlaid_image = overlay_saliency_map(img[0], saliency[0])
    ax2.set_title('Predicted: ' + dataset.classes[class_index])
    # ax2.imshow(saliency[0], cmap=plt.cm.hot)
    ax2.imshow(overlaid_image)
    ax2.axis('off')
    # Save the image
    plt.savefig('saliency_map.png')
    plt.show()
    # Print predicted class
    print('Predicted Class: ' + str(loss))

def gradcam():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = CNN_NeuralNet(3, 5).to(device)
    model = ResNet(5).to(device)
    print(model)
    model.load_state_dict(torch.load('model_bestsweep.pth'))
    # model.load_state_dict(torch.load('model_resnet_stable.pth'))
    model.eval().to(device)
    target_layers = [model.resnet.layer4[-1].conv2]
    cam = CAM(model=model, target_layers=target_layers)
    # Define the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    torch.manual_seed(0)
    dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/archive/CCMT_FInal Dataset', transform)
    with open(f'/home/aditya/Code/interpretable_ml/project/stratified_test_indices.txt', 'r') as f:
        stratified_test_indices = [int(line.rstrip('\n')) for line in f]
    dataset = torch.utils.data.Subset(dataset, stratified_test_indices)
    label="Tomato"
    print(f"Len of dataset: {len(dataset)}")
    if not os.path.exists('saliency_maps_correct'):
        os.makedirs('saliency_maps_correct')
    if not os.path.exists('saliency_maps_incorrect'):
        os.makedirs('saliency_maps_incorrect')
    for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
        # img,label = dataset[250]
        og_img = img.clone()
        label = torch.tensor([dataset.dataset.classes.index(label)]).to(device)
        img = img.unsqueeze(0).to(device)
        input_tensor = img
        input_tensor = input_tensor.to(device)
        output = torch.softmax(model(input_tensor), dim=1)
        pred = torch.argmax(output, dim=1)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred)])
        grayscale_cam = grayscale_cam[0, :]
        img = img[0].cpu().numpy().transpose(1, 2, 0)
            ## Visualize input image and saliency map
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # Title
        ax1.set_title('GT: ' + dataset.dataset.classes[label])

        # Show input image
        ax1.imshow(transforms.ToPILImage()(og_img))
        ax1.axis('off')

        # Overlay the saliency map over the input image
        visualization = show_cam_on_image(img, grayscale_cam)
        ax2.set_title('Predicted: ' + dataset.dataset.classes[pred])
        ax2.imshow(visualization, cmap=plt.cm.hot)
        ax2.axis('off')
        # Save the image
        if pred == label:
            plt.savefig(f'saliency_maps_correct/{idx}.png')
        else:
            plt.savefig(f'saliency_maps_incorrect/{idx}.png')
        # plt.show()
        # Print predicted class
        print('Predicted Class: ' + dataset.dataset.classes[pred])

if __name__ == '__main__':
    # saliency_map()
    gradcam()