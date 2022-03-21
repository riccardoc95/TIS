import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def persistent_entropy(L):
    SL = L.sum()
    HL = - ((L / SL) * np.log(L / SL)).sum()
    return HL

def ps_entropy(lifetime):
    idxs_ = np.flip(np.argsort(lifetime))
    L = lifetime
    HL = persistent_entropy(L)

    LL = L.copy()

    HL0 = HL
    idxs = []
    for i in range(0, idxs_.size):
        l = L[i:].sum()/np.exp(persistent_entropy(L[i:]))
        LL[i] = l
        HL1 = persistent_entropy(LL)
        HLrel = (HL1 - HL0)/(np.log(idxs_.size) - HL)
        HL1 = HL0
        if HLrel > i/(idxs_.size):
            idxs.append(idxs_[i])
        else:
            break

    idxs = np.array(idxs)
    return idxs

def gaussian_mixture(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        idxs = np.where(gm.predict(X) == 1)[0]
    return idxs

def bayesian_gaussian_mixture(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        gm = BayesianGaussianMixture(n_components=2, random_state=0).fit(X)
        idxs = np.where(gm.predict(X) == 1)[0]
    return idxs

def kmeans(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        clustering = KMeans(n_clusters=2, random_state=0).fit(X)
        m = np.array([lifetime[clustering.labels_ == i].mean() for i in np.unique(clustering.labels_)])
        c = np.where(m == m.min())[0]
        idxs = np.where(clustering.labels_ != c)[0]
    return idxs


def mean_shift(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        b = estimate_bandwidth(X)
        clustering = MeanShift(bandwidth=b).fit(X)
        m = np.array([lifetime[clustering.labels_ == i].mean() for i in np.unique(clustering.labels_)])
        c = np.where(m == m.min())[0]
        idxs = np.where(clustering.labels_ != c)[0]
    return idxs

def spectral(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        clustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(X)
        m = np.array([lifetime[clustering.labels_ == i].mean() for i in np.unique(clustering.labels_)])
        c = np.where(m == m.min())[0]
        idxs = np.where(clustering.labels_ != c)[0]
    return idxs

def agglomerative(lifetime):
    X =np.log(1e-05+lifetime).reshape(-1,1)
    if X.shape[0] < 2:
        idxs = np.where(X)[0]
    else:
        clustering = AgglomerativeClustering().fit(X)
        m = np.array([lifetime[clustering.labels_ == i].mean() for i in np.unique(clustering.labels_)])
        c = np.where(m == m.min())[0]
        idxs = np.where(clustering.labels_ != c)[0]
    return idxs
