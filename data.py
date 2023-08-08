from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import random


class MyDataset(Dataset):
    def __init__(self, group, type, train_list, test_list):
        super(MyDataset, self).__init__()
        
        self.train_list = train_list
        self.test_list = test_list
        self.group = group
        self.type = type

        # normalize = transforms.Normalize(mean=[0.7409, 0.5720, 0.5481],
        #                                  std=[0.1278, 0.1308, 0.1420])
        
        normalize = transforms.Normalize(mean=[0.6015, 0.4501, 0.4230],
                                         std=[0.3394, 0.2834, 0.2840])

        transform = transforms.Compose([
            transforms.Resize(224), 
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        

        self.class_dict = {'BCC': 0, 'BKL': 1, 'MEL': 2, 'NV': 3, 'unknown': 4, 'VASC': 5} #label dictionary
        self.transform = transform

        

    def __len__(self):
        if self.type == "train":
            size = len(self.train_list)
        else:
            size = len(self.test_list)
        # print(self.type, " size: ", size)
        return size

    def __getitem__(self, idx):
        if self.type == "train":
            img_path, label, i = self.train_list[idx]
        else:
            img_path, label, i = self.test_list[idx]
        
        img_path = img_path.replace("\\", "/")

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, self.group, i


