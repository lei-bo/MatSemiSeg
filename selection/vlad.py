import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class VLAD:
    def __init__(self, n_words=64, pca_dim=128):
        self.n_words = n_words
        self.pca = PCA(n_components=pca_dim)
        self.kmeans = KMeans(n_clusters=n_words, n_init=1, verbose=0)

    def fit_pca_kmeans(self, embeddings_sample):
        """
        Given samples of embeddings, fit a PCA to reduce the dimension and a
        kmeans to get the visual words.
        :param embeddings_sample: aggregated random samples of pixel embeddings
        from all the images to be select
        """
        embeddings_sample = self.pca.fit_transform(embeddings_sample)
        self.kmeans.fit(embeddings_sample)

    def get_vlad_repr(self, embeddings):
        """
        Compute VLAD representation for the embeddings of a single image. The
        VLAD representation gives the mean residual vector of the visual words.
        :param embeddings: pixel embeddings of a single image with shape (W*H, embedding_dim)
        :return: VLAD representation with shape (n_words, pca_dim)
        """
        embeddings = self.pca.transform(embeddings)
        predictions = self.kmeans.predict(embeddings)
        cluster_centers = self.kmeans.cluster_centers_
        vlad_repr = np.zeros_like(cluster_centers)
        for i in range(0, predictions.shape[0], 10000):
            residuals = embeddings[i:i+10000, :] - cluster_centers[predictions[i:i+10000], :]
            for c in range(cluster_centers.shape[0]):
                vlad_repr[c, :] += residuals[predictions[i:i+10000]==c, :].sum(axis=0)
        return vlad_repr


if __name__ == '__main__':
    sample = np.random.rand(20000, 512)
    embeddings = np.random.rand(312180, 512)
    vlad = VLAD()
    vlad.fit_pca_kmeans(sample)
    print(vlad.get_vlad_repr(embeddings).shape)
