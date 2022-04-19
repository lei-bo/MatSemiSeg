import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Union, List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CSVSplitDataset(Dataset):
    """
    A dataset class that reads a csv split file containing (name, split) pairs
    to get the images in a specified split or the rest of the images.
    """
    def __init__(self,
                 img_dir: str,
                 split_csv: str,
                 split_num: Union[int, List[int]],
                 reverse: bool = False):
        """
        :param img_dir: the directory of all the images
        :param split_csv: the path of the csv file with (name, split) columns
        :param split_num: an int or a list of int specifying the split number
        :param reverse: if true, the images not in split_num are selected
        """
        self.img_dir = img_dir
        if isinstance(split_num, int): split_num = [split_num]
        df = pd.read_csv(split_csv)
        if reverse:
            self.img_names = list(df['name'][~df['split'].isin(split_num)])
        else:
            self.img_names = list(df['name'][df['split'].isin(split_num)])
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.img_names)


if __name__ == '__main__':
    img_dir = "../data/uhcs/images"
    split_csv = "../data/uhcs/splits/split_6fold.csv"
    dataset = CSVSplitDataset(img_dir, split_csv, [0], reverse=True)
    print(len(dataset))