from __future__ import print_function, division
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DefectClassificationDataset(Dataset):
    def __init__(self, transform, set):
        self.set = set
        self.transform = transform
        if set == 'train':
            self.root_dir = 'data/train/'
            self.image_frame = pd.read_csv('train_images.csv')
        elif set == 'val':
            self.root_dir = 'data/val/'
            self.image_frame = pd.read_csv('val_images.csv')
        else:
            self.root_dir = 'data/test/'
            self.image_frame = pd.read_csv('test_images.csv')

    def __len__(self):
        return len(self.image_frame)
    
    def __getitem__(self, idx):
        image_p = os.path.join(self.root_dir + 'images/', self.image_frame['imagename'][idx])
        mask_p = os.path.join(self.root_dir + 'annotations/', self.image_frame['annotatedname'][idx])
        image = Image.open(image_p)
        image = image.convert('RGB')
        mask = Image.open(mask_p)

        image, mask = self.transform(image, mask)

        return image, mask

