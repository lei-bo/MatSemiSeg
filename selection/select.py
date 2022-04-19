import torch
import numpy as np
import os
from os.path import splitext
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import SpectralClustering

from encoder import HypercolumnVgg
from vlad import VLAD
from optimization import milp_optimize, brute_force_search
from utils import CSVSplitDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Selector:
    def __init__(self,
                 dataset: Dataset,
                 n_select: int,
                 vlad_save_dir: str,
                 n_samples_per_image: int = 20000,
                 n_words: int = 64,
                 pca_dim: int = 128):

        self.dataloader = DataLoader(dataset, batch_size=1)
        self.vlad_save_dir = vlad_save_dir
        self.n_select = n_select
        self.n_samples_per_image = n_samples_per_image
        self.encoder = HypercolumnVgg(n_conv_blocks=3).to(DEVICE)
        self.vlad = VLAD(n_words, pca_dim)

    def select(self):
        """
        Main function to select the most representative images from the given
        dataset. VLAD representations are obtained from direct computation or
        pre-computed.
        """
        if not os.path.exists(vlad_save_dir):
            os.mkdir(vlad_save_dir)
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
        similarities = np.zeros((vlads.shape[0], vlads.shape[0]))
        for i in range(vlads.shape[0]):
            similarities[i, :] = vlads @ vlads[i]
        similarities = similarities + 1
        selected_ids = milp_optimize(similarities, self.n_select)
        print("selected images: ")
        print(img_names[selected_ids])

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
                inputs = inputs.to(DEVICE)
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
                inputs = inputs.to(DEVICE)
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


class Args:
    def __init__(self):
        self.dataset = 'uhcs'
        self.split_csv = 'split_6fold.csv'
        self.test_split = 0
        self.validate_split = 1


if __name__ == '__main__':
    np.random.seed(42)
    args = Args()
    img_dir = f"../data/{args.dataset}/images"
    split_csv_path = f"../data/{args.dataset}/splits/{args.split_csv}"
    vlad_save_dir = f"../data/{args.dataset}/vlads"
    n_samples_per_image = 20000
    pca_dim = 128
    n_words = 64
    n_select = 2

    dataset = CSVSplitDataset(img_dir,
                              split_csv_path,
                              split_num=[args.validate_split, args.test_split],
                              reverse=True)

    selector = Selector(dataset, n_select, vlad_save_dir,
                        n_samples_per_image, n_words, pca_dim)
    selector.select()
