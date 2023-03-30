import torch
import numpy as np
import os
from os.path import splitext
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import SpectralClustering, KMeans
from abc import abstractmethod
from argparse import ArgumentParser

from .encoder import HypercolumnVgg
from .vlad import VLAD
from .optimization import brute_force_search
from .datasets import DatasetTemplate, CSVSplitDataset, TextSplitDataset


class Selector:
    def __init__(self,
                 dataset: DatasetTemplate,
                 n_select: int,
                 vlad_save_dir: str,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128,
                 device: str = 'cuda'):

        self.dataloader = DataLoader(dataset, batch_size=1)
        self.img_names = np.array(dataset.img_names)
        self.vlad_save_dir = vlad_save_dir
        self.n_select = n_select
        self.n_samples_per_image = n_samples_per_image
        self.encoder = HypercolumnVgg(n_conv_blocks=3).to(device)
        self.vlad = VLAD(n_words, pca_dim)
        self.device = device

    def select(self):
        """
        Main function to select the most representative images from the given
        dataset. VLAD representations are obtained from direct computation or
        pre-computed.
        """
        if not os.path.exists(self.vlad_save_dir):
            print(f"VLAD representations not found at {self.vlad_save_dir}. Computing now...")
            os.makedirs(self.vlad_save_dir, exist_ok=True)
            print("--- sample pixel-level hypercolumn embeddings ---")
            samples = self.sample_cnn_embedding()
            print(f"shape of sampled pixel embeddings before PCA and KMeans: {samples.shape}")
            print("--- fitting pca and kmeans ---", end='')
            self.vlad.fit_pca_kmeans(samples)
            print(" finished!")
            print("--- computing VLAD representations ---")
            vlads = self.compute_vlads()
        else:
            print(f"VLAD representations found at {self.vlad_save_dir}")
            vlads = self.load_vlads()
        selected = self.select_from_vlads(vlads)
        print("selected images: ", selected)
        return selected

    @abstractmethod
    def select_from_vlads(self, vlads):
        raise NotImplementedError

    def sample_cnn_embedding(self):
        """
        Sample pixel-level hypercolumn embeddings for each image to get visual
        words after running PCA and kmeans clustering.
        :return: sampled pixel embeddings for the dataset with shape
        (n_sample, feature_dim)
        """
        with torch.no_grad():
            self.encoder.eval()
            embeddings_sample = []
            for inputs, _ in tqdm(self.dataloader):
                assert inputs.shape[0] == 1
                inputs = inputs.to(self.device)
                outputs = self.encoder(inputs)  # shape[1, C, H, W]
                outputs = outputs.squeeze().cpu().numpy()
                embeddings = outputs.reshape(outputs.shape[0], -1).T
                rand_ind = np.random.choice(embeddings.shape[0],
                                            self.n_samples_per_image,
                                            replace=False)
                embeddings = embeddings[rand_ind]
                embeddings_sample.append(embeddings)
        return np.vstack(embeddings_sample)

    def compute_vlads(self):
        vlads = []
        with torch.no_grad():
            for inputs, names in tqdm(self.dataloader):
                inputs = inputs.to(self.device)
                output = self.encoder(inputs)  # shape[1, C, H, W]
                output = output.squeeze().cpu().numpy()
                embeddings = output.reshape(output.shape[0], -1).T
                vlad_repr = self.vlad.get_vlad_repr(embeddings)
                vlad_save_path = f"{self.vlad_save_dir}/{splitext(names[0])[0]}.npy"
                np.save(vlad_save_path, vlad_repr)
                vlad_repr = vlad_repr / np.linalg.norm(vlad_repr)
                vlads.append(vlad_repr.flatten())
        return np.vstack(vlads)

    def load_vlads(self):
        vlads = []
        for _, name in self.dataloader:
            vlad_save_path = f"{self.vlad_save_dir}/{splitext(name[0])[0]}.npy"
            vlad_repr = np.load(vlad_save_path)
            vlad_repr = vlad_repr / np.linalg.norm(vlad_repr)
            vlads.append(vlad_repr.flatten())
        return np.vstack(vlads)


class AMRDSelector(Selector):
    def __init__(self,
                 dataset: DatasetTemplate,
                 n_select: int,
                 vlad_save_dir: str,
                 lam: float = 0.1,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128,
                 device: str = 'cuda',
                 choices_txt_path: str = None):

        super().__init__(dataset, n_select, vlad_save_dir,
                         n_samples_per_image, n_words, pca_dim, device)
        if choices_txt_path:
            img_choices = np.loadtxt(choices_txt_path, dtype=str, ndmin=1)
            self.id_choices = [i for i in range(len(self.img_names))
                               if self.img_names[i] in img_choices]
        else:
            self.id_choices = None
        self.lam = lam

    def select_from_vlads(self, vlads):
        similarities = np.zeros((vlads.shape[0], vlads.shape[0]))
        for i in range(vlads.shape[0]):
            similarities[i, :] = vlads @ vlads[i]
        selected_ids = list(brute_force_search(similarities, 1, self.id_choices))
        for _ in range(self.n_select-1):
            selected_ids.append(self.search_next(similarities, selected_ids))
        return self.img_names[selected_ids]

    def search_next(self, similarities, selected_ids):
        indices = set(np.arange(similarities.shape[0]))
        remainder = list(indices.difference(selected_ids))
        best_score = np.NINF
        selected = None
        for i in remainder:
            if self.id_choices is not None and i not in self.id_choices:
                continue
            sim2selected = similarities[i, selected_ids].mean()
            sim2unselected = similarities[i, remainder].mean()
            score = sim2unselected - self.lam * sim2selected
            if score > best_score:
                best_score = score
                selected = i
        return selected


class RandomSelector:
    def __init__(self, dataset: DatasetTemplate, n_select: int, seed: int=42,
                 choices_txt_path: str = None):
        self.dataloader = DataLoader(dataset, batch_size=1)
        self.img_names = dataset.img_names
        self.n_select = n_select
        if choices_txt_path is not None:
            self.img_choices = np.loadtxt(choices_txt_path, dtype=str, ndmin=1)
        else:
            self.img_choices = None
        np.random.seed(seed)

    def select(self):
        if self.img_choices is not None:
            intersection = set(self.img_names).intersection(set(self.img_choices))
            assert len(intersection) > 0
            first_image = np.random.choice(list(intersection), 1)
            pool = set(self.img_names).difference(first_image)
            selected = np.random.choice(list(pool), self.n_select-1, replace=False)
            selected = np.concatenate([first_image, selected])
        else:
            selected = np.random.choice(self.img_names, self.n_select,
                                        replace=False)

        print("selected images: ", selected)
        return selected


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", default="uhcs", required=True)
    arg_parser.add_argument("--split_file", default="split_cv.csv", required=True)
    arg_parser.add_argument("--csv_split_col", default="split")
    arg_parser.add_argument("--csv_split_num", default=0)
    arg_parser.add_argument("--n_select", type=int, required=True)
    arg_parser.add_argument("--method", default='random',
                            choices=['amrd', 'random'])
    arg_parser.add_argument("--amrd_lam", type=float, default=0.1)
    arg_parser.add_argument("--n_samples_per_image", type=int, default=20000)
    arg_parser.add_argument("--pca_dim", type=int, default=128)
    arg_parser.add_argument("--n_words", type=int, default=64)
    arg_parser.add_argument("--gpu_id", type=int, default=0)
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()
    np.random.seed(args.seed)
    args.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    img_dir = f"./data/{args.dataset}/images"
    split_file_path = f"./data/{args.dataset}/splits/{args.split_file}"
    vlad_save_base = f"./data/{args.dataset}/vlads"

    if args.split_file.endswith('.csv'):
        vlad_save_dir = f"{vlad_save_base}/{args.split_file.split('.')[0]}_" \
                        f"{args.csv_split_col}_{args.csv_split_num}/" \
                        f"{args.n_samples_per_image}_{args.pca_dim}_{args.n_words}"
        dataset = CSVSplitDataset(img_dir, split_file_path,
                                  split_num=args.csv_split_num)
    elif args.split_file.endswith('.txt'):
        vlad_save_dir = f"{vlad_save_base}/{args.split_file.split('.')[0]}/" \
                        f"{args.n_samples_per_image}_{args.pca_dim}_{args.n_words}"
        dataset = TextSplitDataset(img_dir, split_file_path)
    else:
        raise ValueError(f"unsupported split file type: {args.split_file}")

    assert len(dataset) >= args.n_select, "select more than the total number of images"
    if args.method == 'amrd':
        selector = AMRDSelector(dataset, args.n_select, vlad_save_dir, args.amrd_lam,
                                  args.n_samples_per_image, args.n_words,
                                  args.pca_dim, args.device)
        save_path = f"./data/{args.dataset}/splits/amrd_lam{args.amrd_lam}_{args.n_select}-shot.txt"
    elif args.method == 'random':
        selector = RandomSelector(dataset, args.n_select, args.seed)
        save_path = f"./data/{args.dataset}/splits/random_{args.n_select}-shot.txt"
    else:
        raise ValueError(f"unsupported method {args.method}")
    selected = selector.select()
    print("saved at ", save_path)
    np.savetxt(save_path, selected, fmt='%s')
