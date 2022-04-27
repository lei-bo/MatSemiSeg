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
from .optimization import milp_optimize, brute_force_search
from .datasets import CSVSplitDataset, TextSplitDataset


class Selector:
    def __init__(self,
                 dataset: Dataset,
                 n_select: int,
                 vlad_save_dir: str,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128,
                 device = 'cuda'):

        self.dataloader = DataLoader(dataset, batch_size=1)
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
            os.makedirs(self.vlad_save_dir, exist_ok=True)
            print("--- sample pixel-level hypercolumn embeddings ---")
            samples = self.sample_cnn_embedding()
            print(f"shape of sampled pixel embeddings before PCA and KMeans: {samples.shape}")
            print("--- fitting pca and kmeans ---", end='')
            self.vlad.fit_pca_kmeans(samples)
            print(" finished!")
            print("--- computing VLAD representations ---")
            vlads, img_names = self.compute_vlads()
        else:
            vlads, img_names = self.get_vlads_from_save()
        selected = self.select_from_vlads(vlads, img_names)
        print("selected images: ", selected)
        return selected

    @abstractmethod
    def select_from_vlads(self, vlads, img_names):
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
        img_names, vlads = [], []
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
                img_names.append(names[0])
        return np.vstack(vlads), np.array(img_names)

    def get_vlads_from_save(self):
        img_names, vlads = [], []
        for _, name in self.dataloader:
            vlad_save_path = f"{self.vlad_save_dir}/{splitext(name[0])[0]}.npy"
            vlad_repr = np.load(vlad_save_path)
            vlad_repr = vlad_repr / np.linalg.norm(vlad_repr)
            vlads.append(vlad_repr.flatten())
            img_names.append(name[0])
        return np.vstack(vlads), np.array(img_names)


class MaxSimSelector(Selector):
    def __init__(self,
                 dataset: Dataset,
                 n_select: int,
                 vlad_save_dir: str,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128,
                 solver: str = "MILP",
                 device = 'cuda'):

        super().__init__(dataset, n_select, vlad_save_dir,
                         n_samples_per_image, n_words, pca_dim, device)
        self.solver = solver

    def select_from_vlads(self, vlads, img_names):
        similarities = np.zeros((vlads.shape[0], vlads.shape[0]))
        for i in range(vlads.shape[0]):
            similarities[i, :] = vlads @ vlads[i]
        similarities = similarities + 1
        if self.solver == "MILP":
            selected_ids = milp_optimize(similarities, self.n_select)
        else:
            selected_ids = brute_force_search(similarities, self.n_select)
        return img_names[selected_ids]


class ClusterSelector(Selector):
    def __init__(self,
                 dataset: Dataset,
                 n_select: int,
                 vlad_save_dir: str,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128,
                 method: str = "spectral",
                 device = 'cuda'):

        super().__init__(dataset, n_select, vlad_save_dir,
                         n_samples_per_image, n_words, pca_dim, device)
        if method == "spectral":
            self.cluster = SpectralClustering(n_clusters=n_select)
        elif method == "kmeans":
            self.cluster = KMeans(n_clusters=n_select)
        else:
            raise NotImplementedError(method)

    def select_from_vlads(self, vlads, img_names):
        cluster_ids = self.cluster.fit_predict(vlads)
        selected_ids = []
        for i in range(self.n_select):
            img_ids = np.where(cluster_ids == i)[0]
            vlads_i = vlads[img_ids, :]
            sims = vlads_i @ vlads_i.T
            selected_id = img_ids[np.argmax(sims.sum(axis=1))]
            selected_ids.append(selected_id)
        print(img_names[selected_ids])


class RandomSelector:
    def __init__(self, dataset: Dataset, n_select: int, seed: int=42):
        self.dataloader = DataLoader(dataset, batch_size=1)
        self.n_select = n_select
        np.random.seed(seed)

    def select(self):
        img_names = []
        for _, names in self.dataloader:
            img_names.append(names[0])
        selected = np.random.choice(img_names, self.n_select, replace=False)
        print("selected images: ", selected)
        return selected


class Args:
    def __init__(self):
        self.dataset = 'uhcs'
        self.split_csv = 'split_cv.csv'
        self.test_split = 0
        self.validate_split = 1


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--dataset", default="uhcs", required=True)
    arg_parser.add_argument("--split_file", default="split_cv.csv", required=True)
    arg_parser.add_argument("--csv_split_col", default="split")
    arg_parser.add_argument("--csv_split_num", default=0)
    arg_parser.add_argument("--n_select", type=int, required=True)
    arg_parser.add_argument("--method", default='sim',
                            choices=['sim', 'cluster', 'random'])
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
    if args.method == 'sim':
        selector = MaxSimSelector(dataset, args.n_select, vlad_save_dir,
                                  args.n_samples_per_image, args.n_words,
                                  args.pca_dim, "MILP", args.device)
    elif args.method == 'cluster':
        selector = ClusterSelector(dataset, args.n_select, vlad_save_dir,
                                   args.n_samples_per_image, args.n_words,
                                   args.pca_dim, "kmeans", args.device)
    elif args.method == 'random':
        selector = RandomSelector(dataset, args.n_select, args.seed)
    else:
        raise ValueError(f"unsupported method {args.method}")
    selector.select()
