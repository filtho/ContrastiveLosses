
import numpy as np
from sklearn.manifold import TSNE
from openTSNE import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import scipy
import pandas as pd
from umap import UMAP
import pickle
from pysnptools.snpreader import Bed
import alphashape

import seaborn as sns
from sklearn.metrics import f1_score
import plotly.graph_objects as go
from matplotlib.colors import rgb2hex

def knn_f1_tt(label, coord,tind, vind, k_vec, verbose = True):
    nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord[tind,:])
    #nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord)

    f1_scores = []
    for k in k_vec:
        nbr = nbrs.kneighbors(coord)[1][:, 1:k + 1]
        nbr_labels = np.take(label[tind], nbr)
        #nbr_labels = np.take(label, nbr)
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
        acc = np.sum(predicted_labels[vind] ==label[vind] ) / len(label[vind])
        acc2 = np.sum(predicted_labels[tind] == label[tind]) / len(label[tind])
        acc3 = np.sum(predicted_labels == label) / len(label)
        if verbose: print(f"train acc: {acc2}, valid acc: {acc}, full acc = {acc3}")
    return acc2, acc, acc3
def knn_f1_top2(label, coord, k_vec, return_matches = False):
    nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord)

    f1_scores = []
    for k in k_vec:
        nbr = nbrs.kneighbors(coord)[1][:, 1:k + 1]
        nbr_labels = np.take(label, nbr)
        lst2 = np.empty([len(nbr_labels), 2], dtype=object)
        for i in range(len(nbr_labels)):
            sample = nbr_labels[i, :]
            unique, indexes, counts = np.unique(sample, return_index=True, return_counts=True)
            unique_sorted = [sample[index] for index in sorted(indexes)]

            if len(unique) > 1:
                unique[np.argsort(-counts)][:2]
                lst2[i, :] = unique[np.argsort(-counts)][:2]
            else:
                lst2[i, :] = np.array([unique[np.argmax(counts)], unique[np.argmax(counts)]])
        predicted_labels = np.array(lst2)
        np.where(predicted_labels[:,0] == label)
        f1_score_avg = f1_score(y_true=label, y_pred=predicted_labels[:,0] , average="micro")
        f1_scores.append(f1_score_avg)
        acc = np.sum(np.sum((predicted_labels ==label[:,np.newaxis] ),axis = 1)!=0 )  / len(label)
    if return_matches:
        return f1_scores, acc, np.sum((predicted_labels ==label[:,np.newaxis] ),axis = 1)!=0
    else:
        return f1_scores, acc

def knn_f1(label, coord, k_vec, return_matches = False):
    nbrs = NearestNeighbors(n_neighbors=np.max(k_vec) + 1, algorithm='ball_tree').fit(coord)

    f1_scores = []
    accs = []
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
        np.where(predicted_labels == label)
        f1_score_avg = f1_score(y_true=label, y_pred=predicted_labels, average="micro")
        f1_scores.append(f1_score_avg)
        acc = np.sum(predicted_labels ==label ) / len(label)
        accs.append(acc)

    if return_matches:
        return f1_scores, accs, predicted_labels == label
    else:
        return f1_scores, accs
def latlong(x, y, z):

    lat = np.arcsin(z / (np.sqrt(x ** 2 + y ** 2 + z ** 2)))  # (z / R)
    long = np.arctan2(y, x)

    return lat, long
def EqualEarth(lat, long):
    lat = np.mod(lat+np.pi/2, np.pi) - np.pi/2
    long = np.mod(long+np.pi, np.pi*2) - np.pi

    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    theta = np.arcsin(np.sqrt(3) / 2 * np.sin(lat))
    x = 2 * np.sqrt(3) * long * np.cos(theta) / (
                3 * (9 * A4 * theta ** 8 + 7 * A3 * theta ** 6 + 3 * A2 * theta ** 2 + A1))
    y = A4 * theta ** 9 + A3 * theta ** 7 + A2 * theta ** 3 + A1 * theta
    return x, y

    x, y = EqualEarth(lat, long)


def plotly_plot(quant,coordinates,tind=None, vind=None, knn_fails=False,k = 3, top2 = False, plot_border = False, name = None):
    color_list = [["brown"],
                  ["purple"],
                  ["darkgreen"],
                  ["gold"],
                  ["olive"],
                  ["chocolate"],
                  ["orange"],
                  ["blue"],
                  ["darkgoldenrod"],
                  ["lime"],
                  ["sienna"],
                  ["olivedrab"],
                  ["cyan"],
                  ["black"]]

    color_list = ['black','green', 'orange','magenta','gray',  'red', 'cyan', 'yellow', 'blue',  'pink',  'purple']
    color_list = [
        sns.color_palette("Greys_r", 1),
        sns.color_palette("Greens_r", 1),
        sns.color_palette("Oranges_r", 1),
        # sns.color_palette("PiYG", num_colors_per_superpop * 2),
        sns.color_palette("Blues", 1),
        sns.color_palette("PRGn", 1 ),
        # sns.color_palette("BrBG", num_colors_per_superpop),
        sns.color_palette("Reds", 1),
        sns.color_palette("YlOrRd",1),
        sns.cubehelix_palette(1, reverse=False),
        sns.light_palette("purple", 1 , reverse=True),
        sns.light_palette("navy", 1 , reverse=True),
        sns.light_palette("green", 1 , reverse=True),
        sns.light_palette((210, 90, 60), 1 , reverse=True, input="husl"),
        sns.light_palette("olive", 1 , reverse=True),
        sns.light_palette("darkslategray", 1 , reverse=True),
        sns.light_palette("mediumaquamarine", 1, reverse=True),
        sns.dark_palette("indigo", 1 , reverse=True),
        sns.dark_palette("saddlebrown", 1, reverse=True),
        sns.dark_palette("maroon", 1, reverse=True),
        sns.dark_palette("gold", 1, reverse=True),
        sns.light_palette("lime", 1 , reverse=True),
        sns.light_palette("orange", 1 , reverse=True),

    ]
    #color_list = np.array(color_list).reshape(-1)
    #color_list = [rgb2hex(color_list[i]) for i in range(len(color_list))]


    edge_list = ["black", "red", "white"]
    t2list = np.unique(quant)
    size_list = [8, 3]
    symbol_list = ["circle", "x"]
    jj = 0
    if top2:
        wins = knn_f1_top2(quant, coordinates, [k], return_matches=True)
    else:
        wins = knn_f1(quant, coordinates, [k], return_matches=True)
    if tind is not None:
        ind_list = [tind, vind]
    elif knn_fails:
        #wins = knn_f1(quant, coordinates,[k], return_matches=True)

        ind_list = [np.where(wins[-1])[0], np.where(~wins[-1])[0]]

    else:
        ind_list = [np.arange(len(quant))]
    wind_list = [np.where(wins[-1])[0], np.where(~wins[-1])[0]]

    fig = go.Figure()
    for count2, inds in enumerate(ind_list):
        coords = coordinates[inds, :]
        qq = quant[inds]

        for count, i in enumerate(np.unique(quant)):
            idd = np.where(quant[inds] == i)[0]
            cc = color_list[count % len(color_list)][0]
            'rgba(135, 206, 250, 0.5)'
            color = f'rbg({int(np.round(cc[0] * 256))},{0*int(np.round(cc[1] * 256))},{0*int(np.round(cc[2] * 256))})'
            #color = 'rgb(255,0,0)'
            if coords.shape[-1] ==3:
                  fig.add_trace(
                    go.Scatter3d(
                        mode='markers',
                        x=coords[idd, 0],
                        y=coords[idd, 1],
                        z=coords[idd, 2],
                        hovertext = quant[inds][idd],
                        name = i,
                        marker=dict(
                            symbol = symbol_list[count2],
                            size=size_list[count2],
                            color=rgb2hex(cc),
                            line=dict(
                                color=edge_list[count2],
                                width=1,
                            )
                        ),
                        showlegend=True
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        mode='markers',
                        x=coords[idd, 0],
                        y=coords[idd, 1],
                        hovertext=quant[inds][idd],
                        name=i,

                        # labels= {"pop":quant[inds][idd]},
                        marker=dict(
                            symbol=symbol_list[count2],
                            size=15,
                            #color=color_list[count % len(color_list)],
                            #color=color,#color_list[count % len(color_list)],
                            color = rgb2hex(cc),
                            line=dict(
                                color=edge_list[count2],
                                width=1,
                            )
                        ),

                        showlegend=True
                    )
                )

    def axes_style3d(bgcolor="rgb(0, 0, 0)",
                     gridcolor="rgb(0, 0, 0)",
                     zeroline=False):
        return dict(showbackground=True,
                    #backgroundcolor=bgcolor,
                    #gridcolor=gridcolor,
                    zeroline=False,
                    showticklabels=False,
                    title="",
                    showspikes=False)

    my_axes = axes_style3d()  # black background color


    s1 = np.where(~wins[-1])[0]
    #s2 = np.where(~knn_f1(spops, X_embedded, [3], return_matches=True)[-1])[0]
    #int = np.intersect1d(s1, s2)
    intt = s1
    if coordinates.shape[-1] ==3 and tind is not None:
        fig.add_trace(
            go.Scatter3d(
                mode='markers',
                x=coordinates[intt, 0],
                y=coordinates[intt, 1],
                z=coordinates[intt, 2],
                hovertext=quant[intt],
                marker=dict(
                    symbol="square-open",
                    size=10,
                    color=color_list[count % len(color_list)],
                    line=dict(
                        color="yellow",
                        width=5,)),showlegend=True))

        """ fig.add_trace(
            go.Scatter3d(
                mode='markers',
                x=coordinates[s2, 0],
                y=coordinates[s2, 1],
                z=coordinates[s2, 2],
                hovertext=quant[s2],
                marker=dict(
                    symbol="circle-open",
                    size=10,
                    color="red",
                    line=dict(
                        color="red",
                        width=5,)),showlegend=False))"""
    elif coordinates.shape[-1] ==2 and tind is not None:
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=coordinates[intt, 0],
                y=coordinates[intt, 1],
                #z=coordinates[int, 2],
                hovertext=quant[intt],
                marker=dict(
                    symbol="square-open",
                    size=20,
                    color=color_list[count % len(color_list)],
                    line=dict(
                        color="yellow",
                        width=5, )), showlegend=True))
        """ fig.add_trace(
            go.Scatter(
                mode='markers',
                x=coordinates[s2, 0],
                y=coordinates[s2, 1],
                #z=coordinates[s2, 2],
                hovertext=quant[s2],
                marker=dict(
                    symbol="circle-open",
                    size=20,
                    color="red",
                    line=dict(
                        color="red",
                        width=5, )), showlegend=False))"""
        fig.update(layout_xaxis_range=[1.1*np.min(coordinates[:,0]),1.1*np.max(coordinates[:,0])])
        fig.update(layout_yaxis_range=[1.1*np.min(coordinates[:,1]),1.1*np.max(coordinates[:,1])])
    fig.update_layout(
        title = f"knn score {wins[1]}, missed {len(np.where(~wins[-1])[0])}, intersection with t-sne: {len(intt)}",
        scene=dict(
            xaxis=my_axes,
            yaxis=my_axes,
            zaxis=my_axes,
            bgcolor='white'
        )
    )


    fig.update_layout(
        title=f"knn score {wins[1]}, missed {len(np.where(~wins[-1])[0])}",
        scene=dict(
            xaxis=my_axes,
            yaxis=my_axes,
            zaxis=my_axes,
            bgcolor='white'
        )
    )

    if plot_border:
        # draw sphere
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        latlong(np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1))
        lat, long = latlong(np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1))
        x, y = EqualEarth(lat, long)
        shape = alphashape.alphashape(list(zip(x, y)), alpha=2)
        borderx, bordery = shape.exterior.coords.xy
        fig.add_trace( go.Scatter(x = np.array(borderx), y =  np.array(bordery), mode="lines"))

    if name==None:
        name = "test3"
    fig.write_html(f"{name}.html")
    fig.show()



num_colors_per_superpop = 6

fam = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam", header = None, delimiter = " ").to_numpy()
bim = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.bim", header = None, delimiter = "\t").to_numpy()
spop = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations", header = None).to_numpy()
spopped = [spop[np.where(i == spop[:,0])[0],1][0] for i in fam[:,0]]
spops = np.array(spopped)

fam_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam", header = None, delimiter = " ").to_numpy()
bim_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.bim", header = None, delimiter = "\t").to_numpy()

spop_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations", header = None).to_numpy()
spopped_human = [spop_human[np.where(i == spop_human[:,0])[0],1][0] for i in fam_human[:,0]]
spops_human = np.array(spopped_human)

train_inds = pd.read_csv("train_index_dog.csv", header=None, delimiter=",").to_numpy().flatten()
valid_inds = pd.read_csv("valid_index_dog.csv", header=None, delimiter=",").to_numpy().flatten()

train_inds_human = pd.read_csv("train_index_human.csv", header=None, delimiter=",").to_numpy().flatten()
valid_inds_human = pd.read_csv("valid_index_human.csv", header=None, delimiter=",").to_numpy().flatten()

# File handles needed for kaus plot.
proj_fam_file_dog = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam"
spop_file_dog  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations"
proj_fam_file_human = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam"
spop_file_human  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations"



# DOG DATASET
geno_data = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_filtered.parquet").to_numpy()
mcg = scipy.stats.mode(geno_data,axis = 1)
idx = np.where(geno_data == 9 )
geno_data[idx] = mcg.mode[idx[0]]
geno_data_normed  = (geno_data - np.mean(geno_data[:,train_inds], axis = 1, keepdims = True)) / np.std(geno_data[:,train_inds], axis = 1, keepdims = True)

#Human data
geno_data_human  = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.parquet").to_numpy()
mcg = scipy.stats.mode(geno_data_human,axis = 1)
idx = np.where(geno_data_human == 9 )
geno_data_human[idx] = mcg.mode[idx[0]]
geno_data_human = geno_data_human.astype(np.int8)
geno_data_human_normed = (geno_data_human - np.mean(geno_data_human[:,train_inds_human], axis = 1, keepdims = True)) / np.std(geno_data_human[:,train_inds_human], axis = 1, keepdims = True)
# Human data missing
geno_data_valmiss = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered_valmiss.parquet").to_numpy()
nine = np.sum(geno_data_valmiss == 9, axis = 0) / geno_data_valmiss.shape[0]
mcg = scipy.stats.mode(geno_data_valmiss,axis = 1)
idx = np.where(geno_data_valmiss == 9 )
geno_data_valmiss[idx] = mcg.mode[idx[0]]#[:,0]
geno_data_valmiss_normed  = (geno_data_valmiss - np.mean(geno_data_valmiss[:,train_inds_human], axis = 1, keepdims = True)) / np.std(geno_data_valmiss[:,train_inds_human], axis = 1, keepdims = True)

# 2D PCA Embeddings:
pca = PCA(n_components=2)
pca.fit(geno_data_normed[:,train_inds].T)
X_PCA = pca.transform(geno_data_normed.T)
#np.save("saved_files/PCA_dog.npy", X_PCA)

pca = PCA(n_components=2)
pca.fit(geno_data_human_normed[:,train_inds_human].T)
X_PCA_human = pca.transform(geno_data_human_normed.T)
#np.save("saved_files/PCA_human.npy", X_PCA_human)

pca = PCA(n_components=2)
pca.fit(geno_data_valmiss_normed[:,train_inds_human].T)
X_PCA_human_valmiss = pca.transform(geno_data_valmiss_normed.T)
#np.save("saved_files/PCA_human_masked.npy", X_PCA_human_valmiss)

# 100D PCA embeddings for PCA-preprocessing
pca = PCA(n_components=100)
pca.fit(geno_data_human_normed[:,train_inds_human].T)
X_PCA100_human = pca.transform(geno_data_human_normed.T)

pca = PCA(n_components=100)
pca.fit(geno_data_valmiss_normed[:,train_inds_human].T)
X_PCA100_human_valmiss = pca.transform(geno_data_valmiss_normed.T)

pca = PCA(n_components=100)
pca.fit(geno_data_normed[:,train_inds].T)
X_PCA100_dog = pca.transform(geno_data_normed.T)

# Define train/predict UMAP and t-SNE functions.
def umapper(n, data, train_inds_,valid_inds_):
    umap_train = UMAP(n_neighbors=n, min_dist=0.1, spread=1, n_jobs=-1, verbose=True).fit(data[:, train_inds_].T)
    # Transform the valid samples
    umap_valid = umap_train.transform(data[:, valid_inds_].T)
    # Concatenate them, same ordering.
    umapped = np.zeros([data.shape[1], 2])
    umapped[train_inds_, :] = umap_train.embedding_
    umapped[valid_inds_, :] = umap_valid
    return umapped
def tsne_mapper(n, data, train_inds_,valid_inds_):
    coords_tsne = TSNE(n_components=2, perplexity=n, learning_rate="auto", initialization='random', verbose=True).fit(data[:, train_inds_].T)
    tsne_valid = coords_tsne.transform(data[:, valid_inds_].T)
    tsne_emb = np.zeros([data.shape[1], 2])
    tsne_emb[train_inds_, :] = coords_tsne
    tsne_emb[valid_inds_, :] = tsne_valid
    return tsne_emb


# COMPUTE PCA-PREPROCESSING EMBEDDINGS
# DOG:
for i in [3,30]:
    # DOG UMAP and tsne
    umapped= umapper(i, X_PCA100_dog.T, train_inds, valid_inds)
    #np.save(f"umap_tsne_pca/umap_dog{i}.npy", umapped)
    tsne_mapped= tsne_mapper(i, X_PCA100_dog.T, train_inds, valid_inds)
    #np.save(f"umap_tsne_pca/tsne_dog{i}.npy", tsne_mapped)

for i in [3,30]:
    # Human UMAP and tsne
    umapped= umapper(i, X_PCA100_human.T, train_inds_human, valid_inds_human)
    #np.save(f"saved_files/umap_tsne_pca/umap_human{i}.npy", umapped)
    tsne_mapped= tsne_mapper(i, X_PCA100_human.T, train_inds_human, valid_inds_human)
    #np.save(f"saved_files/umap_tsne_pca/tsne_human{i}.npy", tsne_mapped)

for i in [3,30]:
    # Human Missing UMAP and tsne
    umapped= umapper(i, X_PCA100_human_valmiss.T, train_inds_human, valid_inds_human)
    #np.save(f"saved_files/umap_tsne_pca/umap_human_valmiss{i}.npy", umapped)
    tsne_mapped= tsne_mapper(i, X_PCA100_human_valmiss.T, train_inds_human, valid_inds_human)
    #np.save(f"saved_files/umap_tsne_pca/tsne_human_valmiss{i}.npy", tsne_mapped)


# COMPUTE DISTANCE MATRICES

geno_data = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_filtered.parquet").to_numpy()
mcg = scipy.stats.mode(geno_data,axis = 1)
idx = np.where(geno_data == 9 )
geno_data[idx] = mcg.mode[idx[0]]

dd = np.zeros([geno_data.shape[1],geno_data.shape[1]])
for i in range(geno_data.shape[1]):
    for j in range(i,geno_data.shape[1]):
        diff = np.abs(geno_data[:, i] - geno_data[:, j])
        dd[i,j] = np.sum(diff)
    print(i)

dd = dd+dd.T
#np.save("saved_files/distance_dog", dd)

# HUMAN
geno_data_human  = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.parquet").to_numpy()
mcg = scipy.stats.mode(geno_data_human,axis = 1)
idx = np.where(geno_data_human == 9 )
geno_data_human[idx] = mcg.mode[idx[0]]
geno_data_human = geno_data_human.astype(np.int8)
geno_data_human_normed  = (geno_data_human - np.mean(geno_data_human, axis = 1, keepdims = True)) / np.std(geno_data_human, axis = 1, keepdims = True)

dd = np.zeros([geno_data_human.shape[1],geno_data_human.shape[1]])
for i in range(geno_data_human.shape[1]):
    for j in range(i,geno_data_human.shape[1]):
        diff = np.abs(geno_data_human[:, i] - geno_data_human[:, j])
        dd[i,j] = np.sum(diff)
    print(i)
dd = dd+dd.T

np.save("saved_files/distance_human", dd)

# distance matrix for the unfiltered dataset
fam_unfiltered = pd.read_csv("/home/filip/Downloads/NearEastPublic/test.fam", header = None, delimiter = " ").to_numpy()
bim_unfiltered = pd.read_csv("/home/filip/Downloads/NearEastPublic/test.bim", header = None, delimiter = "\t").to_numpy()
test_bed = Bed("/home/filip/Downloads/NearEastPublic/test.bed").read().val
testi = [np.where(i == fam_unfiltered[:,1])[0][0] for i in fam_human[:,1]]

fam_unfiltered2 = fam_unfiltered[testi,:]


dd = np.zeros([test_bed.shape[1],test_bed.shape[1]])
for i in range(test_bed.shape[1]):
    for j in range(i,test_bed.shape[1]):
        diff = np.abs(test_bed[:, i] - test_bed[:, j])
        dd[i,j] = np.sum(diff)
    print(i)

#np.save("saved_files/distance_human_unfiltered", dd)


test_bed[np.isnan(test_bed)] = 9
test_bed = test_bed.astype(np.int8)
mcg = scipy.stats.mode(test_bed,axis = 1)
idx = np.where(test_bed == 9 )
test_bed[idx] = mcg.mode[idx[0]]#[:,0]
test_bed = test_bed.T

test_bed_normed  = (test_bed - np.mean(test_bed[:,train_inds_human], axis = 1, keepdims = True)) / np.std(test_bed[:,train_inds_human], axis = 1, keepdims = True)

pca = PCA(n_components=2)
pca.fit(test_bed_normed[:,train_inds_human].T)
X_PCA_test = pca.transform(test_bed_normed.T)
X_PCA_test = X_PCA_test[testi, :]
np.save("saved_files/PCA_unfiltered_human.npy", X_PCA_test)

plotly_plot(spops_human, X_PCA_test)

PCA_human = np.load("saved_files/PCA_human.npy")


knn_f1_tt(fam_human[:,0], X_PCA_test, train_inds_human, valid_inds_human, [3])
knn_f1_tt(fam_human[:,0], PCA_human, train_inds_human, valid_inds_human, [3])
def global_spop(subpops_, spops_, embedding):
    # GLOBAL-ISH - Collapsing samples to mean of the subpopulation, and computing 3NN  superpopulation classification

    _, idx = np.unique(subpops_, return_index=True)
    crds_embedding = []
    for i in np.unique(subpops_):
        indd = np.where(subpops_ == i)[0]
        crds_embedding.append(np.mean(embedding[indd, :], axis=0))

    pop_embedding = np.concatenate(crds_embedding).reshape((len(crds_embedding), embedding.shape[1]))

    glob_spop_umap = knn_f1(spops_[idx], pop_embedding, k_vec=[3])[0]
    return     glob_spop_umap, pop_embedding

global_spop(fam_human[:,0], spops_human, X_PCA_test)[0]
global_spop(fam_human[:,0], spops_human, PCA_human)[0]

plotly_plot(spops_human, X_PCA_test)
plotly_plot(spops_human, PCA_human)
