import os
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils import data
import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

param = config.get_param()

batch_size = param["training_param"]["batch_size"]

class Haemocytes(Dataset):
    def __init__(self, srcpath):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.srcpath = srcpath
        self.dataset = {"image": [], "label": []}
        for roots, dirs, files in os.walk(self.srcpath):
            for file in files:
                file_path = os.path.join(roots, file)
                label = int(self.get_info(file_path))
                image = cv2.imread(file_path)
                image = cv2.resize(image, (32, 32))
                image = self.transform(image)
                self.dataset['image'].append(image)
                self.dataset['label'].append(label)
        self.dataset['label'] = torch.LongTensor(self.dataset['label'])

    def __getitem__(self, index):
        image, label = self.dataset['image'][index], self.dataset['label'][index]

        return image, label

    def __len__(self):
        return len(self.dataset['image'])

    def get_info(self, file_path):
        label_dir, file_name = os.path.split(file_path)
        hospital_dir, label = os.path.split(label_dir)

        return label


def load_data_one_client(srcpath):
    haemocytes = Haemocytes(srcpath)
    data_len = len(haemocytes)
    train_size = int(data_len * 0.8)
    test_size = data_len - train_size
    trainset, testset = torch.utils.data.random_split(haemocytes, [train_size, test_size])
    trainloader = data.DataLoader(trainset, batch_size, shuffle=False, num_workers=0)
    testloader = data.DataLoader(testset, batch_size, shuffle=False, num_workers=0)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}

    return trainloader, testloader, num_examples






