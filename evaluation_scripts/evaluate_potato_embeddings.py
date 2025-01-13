import pandas as pd
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from xarray import DataArray
import math
from pandas_plink import read_plink1_bin, write_plink1_bin
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.manifold import TSNE
from pandas_plink import read_plink1_bin, write_plink1_bin
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import scipy
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')



## Read data
path = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/potato"
# genotypes
gdat = pd.read_parquet(f"{path}/potato.parquet").to_numpy()

# phenotypes
potphen = pd.read_csv(f"{path}/potato.pheno", delimiter=",")
phn = "TN"
atw3 = potphen[phn].to_numpy()


# FAM file
fam = pd.read_csv(f"{path}/potato.fam", header=None, delimiter=" ").to_numpy()
ids = fam[:, 0]

# Define regression methods
def nearest_neighbour_pheno_tt(quant_train, quant_test, coord_train, coord_test, k_vec):
    nind_train = np.where(~np.isnan(quant_train))[0]
    nind_test = np.where(~np.isnan(quant_test))[0]

    coord_train = coord_train[nind_train, :]
    coord_test = coord_test[nind_test, :]
    nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord_train)
    corr = []
    pearr = []
    quant_train = quant_train[nind_train]
    quant_test = quant_test[nind_test]

    for k in k_vec:
        nbr = nbrs.kneighbors(coord_test)[1][:, 1:k + 1]
        nbr_labels = np.take(quant_train, nbr)
        predicted = np.mean(nbr_labels, axis=1)
        corr.append(np.corrcoef(predicted, quant_test)[0, 1])
        pearr.append(scipy.stats.pearsonr(quant_test, predicted)[0])

    return corr, pearr
def BayesianRidgepheno_tt(quant_train, quant_test, coord_train, coord_test):
    nind_train = np.where(~np.isnan(quant_train))[0]
    nind_test = np.where(~np.isnan(quant_test))[0]
    coord_train = coord_train[nind_train, :]
    coord_test = coord_test[nind_test, :]
    quant_train = quant_train[nind_train]
    quant_test = quant_test[nind_test]

    clf = linear_model.BayesianRidge(tol=1e-7)
    clf.fit(coord_train, quant_train)
    predicted = clf.predict(coord_test)
    corr = clf.score(coord_test, quant_test)
    pearr = scipy.stats.pearsonr(quant_test, predicted)[0]

    mse = np.mean((quant_test - predicted) ** 2)

    return corr, mse, pearr# , predicted
def random_forest_tt(quant_train, quant_test, coord_train, coord_test, depth=20 ):
    nind_train = np.where(~np.isnan(quant_train))[0]
    nind_test = np.where(~np.isnan(quant_test))[0]
    coord_train = coord_train[nind_train, :]
    # coord_train = (coord_train - np.mean(coord_train, axis = 0))/np.std(coord_train, axis = 0)
    coord_test = coord_test[nind_test, :]
    # coord_test = (coord_test - np.mean(coord_train, axis = 0))/np.std(coord_train, axis = 0)

    quant_train = quant_train[nind_train]
    # quant_train = (quant_train - np.mean(quant_train, axis = 0))/np.std(quant_train, axis = 0)
    quant_test = quant_test[nind_test]
    # quant_test = (quant_test - np.mean(quant_test, axis = 0))/np.std(quant_test, axis = 0)

    model = RandomForestRegressor(n_estimators=100, max_features=5, max_depth=depth)
    model.fit(coord_train, quant_train)
    predicted = model.predict(coord_test)
    # plt.plot(quant_test, quant_test, '.')
    # plt.plot(quant_test, predicted, '.')
    # plt.plot(quant_train, model.predict(coord_train), '.')
    # plt.axis("equal")    #print()
    # plt.show()
    # print(model.score(coord_train, quant_train), model.score(coord_test, quant_test))
    corr = model.score(coord_test, quant_test)
    mse = np.mean((predicted - quant_test) ** 2)
    pearr = scipy.stats.pearsonr(quant_test, predicted)[0]

    return corr, mse, pearr #, predicted
def knn_f1(label, coord, k_vec):
    nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord)

    f1_scores = []
    for k in k_vec:
        nbr = nbrs.kneighbors(coord)[1][:, 1:k + 1]
        nbr_labels = np.take(label, nbr)
        lst = []
        for i in range(len(nbr_labels)):
            sample = nbr_labels[i, :]
            unique, indexes, counts = np.unique(sample, return_index=True, return_counts=True)
            unique_sorted = [sample[index] for index in sorted(indexes)]
            if counts[np.argmax(counts)] > 1:
                lst.append(unique[np.argmax(counts)])
            else:
                lst.append(unique_sorted[np.where(unique[indexes] != "nan")[0][0]])
        predicted_labels = np.array(lst)
        f1_score_avg = f1_score(y_true=label, y_pred=predicted_labels, average="micro")
        f1_scores.append(f1_score_avg)
    return f1_scores







# Evaluate methods using N different train/test split realizations.

N = 10
kkn = np.zeros(N)
kkntsne = np.zeros((N, 3))
kknpca = np.zeros((N, 3))
kknfull = np.zeros((N, 3))

f1s_pca10d = np.zeros((N,len(np.arange(3,50))))
f1s_pca = np.zeros((N,len(np.arange(3,50))))
f1s_tsne = np.zeros((N,len(np.arange(3,50))))
k = [10]
y = -1
knn_vecced_tsne = np.zeros((N, len(np.arange(3, 50))))
knn_vecced_pca = np.zeros((N, len(np.arange(3, 50))))
knn_vecced_full = np.zeros((N, len(np.arange(3, 50))))
k_vec = np.arange(3,50)
for i in range(N):
    train_inds, valid_inds, pops_train, pops_test = train_test_split(np.arange(669), np.arange(669), test_size=0.2,
                                                                     stratify=ids, random_state=i)
    pca = PCA(n_components=2)
    pca.fit(gdat.T)
    X_PCA = pca.transform(gdat.T)
    #rand_coords = np.random.uniform(low=-1, high=1, size=np.shape(X_PCA[:, :2]))
    X_emb = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(gdat.T)

    kkntsne[i, 0] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], X_emb[train_inds, :], X_emb[valid_inds, :], k)[y][0]
    kkntsne[i, 1] = BayesianRidgepheno_tt(atw3[train_inds], atw3[valid_inds], X_emb[train_inds, :], X_emb[valid_inds, :], )[y]
    kkntsne[i, 2] = random_forest_tt(atw3[train_inds], atw3[valid_inds], X_emb[train_inds, :], X_emb[valid_inds, :], )[y]

    knn_vecced_tsne[i, :]  = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], X_emb[train_inds, :], X_emb[valid_inds, :], np.arange(3, 50))[-1]

    kknpca[i, 0] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], X_PCA[train_inds, :], X_PCA[valid_inds, :], k)[y][0]
    kknpca[i, 1] = BayesianRidgepheno_tt(atw3[train_inds], atw3[valid_inds], X_PCA[train_inds, :], X_PCA[valid_inds, :], )[y]
    kknpca[i, 2] = random_forest_tt(atw3[train_inds], atw3[valid_inds], X_PCA[train_inds, :], X_PCA[valid_inds, :], )[y]
    knn_vecced_pca[i, :]  = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], X_PCA[train_inds, :], X_PCA[valid_inds, :], np.arange(3, 50))[-1]

    kknfull[i, 0] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], gdat.T[train_inds, :], gdat.T[valid_inds, :], k)[y][0]
    kknfull[i, 1] = BayesianRidgepheno_tt(atw3[train_inds], atw3[valid_inds], gdat.T[train_inds, :], gdat.T[valid_inds, :], )[y]
    kknfull[i, 2] = random_forest_tt(atw3[train_inds], atw3[valid_inds], gdat.T[train_inds, :], gdat.T[valid_inds, :], )[y]
    knn_vecced_full[i, :]  = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], gdat.T[train_inds, :], gdat.T[valid_inds, :], np.arange(3, 50))[-1]

    f1s_pca[i,:]  =  knn_f1(ids, X_PCA[:,:2],k_vec)
    f1s_pca10d[i,:]  =  knn_f1(ids, X_PCA,k_vec)
    f1s_tsne[i,:] = knn_f1(ids, X_emb,k_vec)


mean_tsne = np.mean(knn_vecced_tsne,axis = 0)
mean_pca = np.mean(knn_vecced_pca,axis = 0)
mean_full = np.mean(knn_vecced_full,axis = 0)

mean_pca2 = np.mean(kknpca,axis = 0)
mean_tsne2 = np.mean(kkntsne, axis = 0)
mean_full2 = np.mean(kknfull, axis = 0)


plt.figure()
plt.plot(mean_pca)
plt.plot(mean_tsne)
plt.plot(mean_full)


plt.figure()
plt.plot(mean_pca2, label = "pca")
plt.plot(mean_tsne2, label = "t-SNE")
plt.plot(mean_full2, label = "All SNPs")
plt.ylabel("Pearson Correlation")
plt.title("Genomic prediction of tuber number")
plt.xticks([0,1,2], ["KNN", "GBLUP", "RF" ])
plt.legend()

plt.figure()
plt.plot(np.mean(f1s_tsne, axis = 0), label = "tsne")
plt.plot(np.mean(f1s_pca10d, axis = 0), label = "pca10d")
plt.plot(np.mean(f1s_pca, axis = 0), label = "pca")
plt.legend()
plt.ylabel("Pearson Correlation")



np.savetxt("f1s_tsne_plt.txt", f1s_tsne)
np.savetxt("f1s_pca_plt.txt", f1s_pca)
np.savetxt("f1s_pca10d_plt.txt", f1s_pca10d)


import os
def h5_keys(filename):
    with h5py.File(filename, 'r') as hf:
        keys = list(hf.keys())
    return keys
def read_h5(filename, dataname):
    '''
    Read data from a h5 file.

    :param filename: directory and filename (with .h5 extension) of file to read from
    :param dataname: name of the datataset in the h5
    :return the data
    '''
    with h5py.File(filename, 'r') as hf:
        data = hf[dataname][:]
        # print(hf.keys())
    return data

def eval_send_data(dir,pltnr = 1,k = 5):
    files = np.array(os.listdir(dir))[np.where([f[-1] == "5" for f in os.listdir(dir)])[0]]

    N2 = len(files)
    stats_test = np.zeros((N2,3))
    stats_train = np.zeros((N2,3))
    knn_vecced = np.zeros((N2, len(np.arange(3,50))))
    k = [k]
    f1 = np.zeros((N2, len(np.arange(3,50))))
    for count,file in enumerate(files):
        print(count)
        eps = h5_keys(f"{dir}/{file}")[:-1]
        epochs = np.array([item.split("_")[0] for  item in  eps]).astype(int)
        n = file[0]
        train_inds = np.loadtxt(f"{dir}/{n}_train_index.csv", delimiter = ",").astype(int)
        valid_inds = np.loadtxt(f"{dir}/{n}_valid_index.csv", delimiter = ",").astype(int)
        val_loss = np.loadtxt(f"{dir}/{n}_validation_loss.csv", delimiter = ",")
        smallest_val_index  =np.argmin(np.abs(np.argmin(val_loss[:,1]).astype(int)-epochs))
        #plt.figure(pltnr+1000)
        #plt.plot(val_loss[:,0], val_loss[:,1])
        #plt.vlines(ymin = np.min(val_loss[:,1]), ymax = np.max(val_loss[:,1]), x= np.argmin(val_loss[:,1]), colors = 'k')
        order = np.argsort(np.array([e.split("_")[0] for e in eps]).astype(int))

        # Take last epoch
        i_coords = read_h5(f"{dir}/{file}", np.array(eps)[order[-1]])
        # Take epoch with smallest val loss
        i_coords = read_h5(f"{dir}/{file}", eps[smallest_val_index])

        #i_coords = read_h5(f"{dir}/{file}", '100_encoded_train')
        stats_test[count,0] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], i_coords[train_inds, :],i_coords[valid_inds, :], k)[-1][0]
        stats_train[count,0] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[train_inds], i_coords[train_inds, :],i_coords[train_inds, :], k)[-1][0]

        knn_vecced[count, :] = nearest_neighbour_pheno_tt(atw3[train_inds], atw3[valid_inds], i_coords[train_inds, :], i_coords[valid_inds, :],np.arange(3, 50))[-1]

        stats_test[count,1] = BayesianRidgepheno_tt(atw3[train_inds], atw3[valid_inds], i_coords[train_inds, :2],i_coords[valid_inds, :2])[-1]
        stats_train[count,1] = BayesianRidgepheno_tt(atw3[train_inds], atw3[train_inds], i_coords[train_inds, :2],i_coords[train_inds, :2])[-1]

        stats_test[count,2] = random_forest_tt(atw3[train_inds], atw3[valid_inds], i_coords[train_inds, :2], i_coords[valid_inds, :2])[-1]
        stats_train[count,2] = random_forest_tt(atw3[train_inds], atw3[train_inds], i_coords[train_inds, :2], i_coords[train_inds, :2])[-1]

        f1[count,:] = knn_f1(ids, i_coords, np.arange(3,50))


    return stats_train, stats_test, f1, knn_vecced


full_dict_f1 = {}
# This points to
dir = "/home/filip/Documents/ContrastiveLosses/send_data_full"

# These points to a directory which includes directories of saved runs from run_gcae.py.
dir1 = f"{dir}/send_data0_100_orig_"
dir2 = f"{dir}/send_data0_0_orig_"

stats_train, stats_test100, f1cont2d100, knn_vec_cont2d100= eval_send_data(f"{dir}/send_data0_100_orig_", k = 13)
stats_train, stats_test0, f1cont2d0, knn_vec_cont2d0= eval_send_data(f"{dir}/send_data0_0_orig_")


#np.savetxt("f1cont2d0_plt.txt", f1cont2d0)
#np.savetxt("f1cont2d100_plt.txt", f1cont2d100)


sns.set("poster")
sns.set()

fig, ax1 = plt.subplots(figsize=(15,5))
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels

left, bottom, width, height = [0.2, 0.2, 0.2, 0.3]
tt = 669
ax1.plot(k_vec[:tt], np.mean(f1cont2d0, axis = 0)[:tt], label="Contrastive")
ax1.plot(k_vec[:tt], np.mean(f1cont2d100, axis = 0)[:tt], label="Pheno-informed Contrastive")
ax1.plot(k_vec[:tt], np.mean(f1s_tsne, axis = 0)[:tt], label="t-SNE")
ax1.plot(k_vec[:tt], np.mean(f1s_pca, axis = 0)[:tt], label="PCA 2D")
ax1.plot(k_vec[:tt], np.mean(f1s_pca10d, axis = 0)[:tt], label="PCA 10D")
plt.xlabel("Number of neighbours k")
plt.ylabel("KNN classification f1-score")
ax2 = fig.add_axes([left, bottom, width, height])

tt2 = 60
ax2.plot(k_vec[:tt2], np.mean(f1cont2d0, axis = 0)[:tt2], label="Contrastive")
ax2.plot(k_vec[:tt2], np.mean(f1cont2d100, axis = 0)[:tt2], label="Pheno-informed Contrastive")

ax2.plot(k_vec[:tt2], np.mean(f1s_tsne, axis = 0)[:tt2], label="t-SNE")
ax2.tick_params(color='red', labelcolor='black')
for spine in ax2.spines.values():
    spine.set_edgecolor('red')

ax1.plot([3,50,50,3,3],[0.9,0.9,0.8,0.8,0.9],'r')
#ax1.plot([3,40],[0.8,0.56 ],'r--')
ax1.plot([3,40],[0.8,0.25 ],'r--')
ax1.plot([50,230],[0.9,0.545 ],'r--')
ax1.legend()
ax1.legend(loc=1, prop={'size': 15})



