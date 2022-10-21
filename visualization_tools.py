import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap


def VisualizeFeatureByTSNE(features, labels, n_class, n_dim, figname='t-SNE', color_map='tab20'):
    decomp = TSNE(n_components=n_dim)
    X_decomp = decomp.fit_transform(features)
    cmap = get_cmap(color_map)
    for i in range(n_class):
        # marker = "$" + str(i) + "$"
        marker = '.'
        indices = (labels == i)
        plt.scatter(X_decomp[indices, 0], X_decomp[indices,
                                                   1], marker=marker, color=cmap(i))
    plt.title(f"t-SNE")
    plt.savefig(figname+'.png')
    plt.savefig(figname+'.eps')
    plt.close()


def VisualizeFeatureByPCA(features, labels, n_class, n_dim, figname='PCA', color_map='tab20'):
    decomp = PCA(n_components=n_dim)
    X_decomp = decomp.fit_transform(features)
    cmap = get_cmap(color_map)
    for i in range(n_class):
        # marker = "$" + str(i) + "$"
        marker = '.'
        indices = (labels == i)
        plt.scatter(X_decomp[indices, 0], X_decomp[indices,
                                                   1], marker=marker, color=cmap(i))
    plt.title(f"PCA")
    plt.savefig(figname+'.png')
    plt.savefig(figname+'.eps')
    plt.close()
