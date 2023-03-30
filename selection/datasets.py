import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Union, List


class DatasetTemplate(Dataset):
    """A dataset template class that supports loading and transforming images.
     Subclasses should implement how to get self.img_names.
    """
    def __init__(self, img_dir):
        """
        :param img_dir: the directory where the images are stored
        """
        self.img_dir = img_dir
        self.img_names = []
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


class CSVSplitDataset(DatasetTemplate):
    """
    A dataset class that reads a csv split file containing (name, split) pairs
    to get the images in a specified split or the rest of the images.
    """
    def __init__(self,
                 img_dir: str,
                 split_csv: str,
                 split_num: Union[int, List[int]],
                 split_col_name: str = "split",
                 reverse: bool = False):
        """
        :param split_csv: the path of the csv file containing the split information
        :param split_num: an int or a list of int specifying the split number
        :param split_col_name: the name of the column containing the split number
        :param reverse: if true, the images not in split_num are selected
        """
        super().__init__(img_dir)
        if isinstance(split_num, int): split_num = [split_num]
        df = pd.read_csv(split_csv)
        if reverse:
            self.img_names = list(df['name'][~df[split_col_name].isin(split_num)])
        else:
            self.img_names = list(df['name'][df[split_col_name].isin(split_num)])


class TextSplitDataset(DatasetTemplate):
    """A dataset class that reads a text split file containing the name of the
    images in the target dataset split.
    """
    def __init__(self, img_dir: str, split_txt: str):
        """
        :param split_txt: the path of the text file that contains the names of
        the images in the split
        """
        super().__init__(img_dir)
        self.img_names = np.loadtxt(split_txt, dtype=str, delimiter='\n', ndmin=1)


if __name__ == '__main__':
    img_dir = "./data/uhcs/images"
    split_csv = "./data/uhcs/splits/split_cv.csv"
    dataset = CSVSplitDataset(img_dir, split_csv, [0], reverse=True)
    print(len(dataset))
    split_csv = "./data/uhcs/splits/train16A.txt"
    dataset = TextSplitDataset(img_dir, split_csv)
    print(len(dataset))