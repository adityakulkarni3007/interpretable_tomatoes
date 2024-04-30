import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import numpy as np

class TomatoDataset(Dataset):
    '''
    structure of the dataset
    img_dir
        ├── Tomato___class_name
        │   ├── image1.jpg
    '''
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.classes = os.listdir(img_dir)
        print(f"Len of classes: {len(self.classes)}")
        self.img_paths = []
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(img_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_paths.append(img_path)
        # np.random.shuffle(self.img_paths)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path)
        img.resize((224,224), Image.BICUBIC)
        label = img_path.split('/')[-2]
        if self.transform:
            img = self.transform(img)
        return img, label
    
if __name__ == '__main__':
    dataset = TomatoDataset('/home/aditya/Code/interpretable_ml/project/archive/CCMT_FInal Dataset')
    print(len(dataset))
    for i in range(len(dataset)):
        img, label = dataset[i]
        img = np.array(img)
        if img.shape[-1]!=3:
            print(img.shape)
            print(label)