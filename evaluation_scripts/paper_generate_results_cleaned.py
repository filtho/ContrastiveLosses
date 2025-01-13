import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alphashape
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
import h5py
import scipy
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import rgb2hex

num_colors_per_superpop = 6

fam = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam", header = None, delimiter = " ").to_numpy()
bim = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.bim", header = None, delimiter = "\t").to_numpy()
spop = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations", header = None).to_numpy()
spopped = [spop[np.where(i == spop[:,0])[0],1][0] for i in fam[:,0]]
spops = np.array(spopped)

fam_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam", header = None, delimiter = " ").to_numpy()
bim_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.bim", header = None, delimiter = "\t").to_numpy()

spop_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations", header = None).to_numpy()
spop_human = pd.read_csv("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations", header = None).to_numpy()
spopped_human = [spop_human[np.where(i == spop_human[:,0])[0],1][0] for i in fam_human[:,0]]
spops_human = np.array(spopped_human)

train_inds = pd.read_csv("train_index_dog.csv", header=None, delimiter=",").to_numpy().flatten()
valid_inds = pd.read_csv("valid_index_dog.csv", header=None, delimiter=",").to_numpy().flatten()

train_inds_human = pd.read_csv("train_index_human.csv", header=None, delimiter=",").to_numpy().flatten()
valid_inds_human = pd.read_csv("valid_index_human.csv", header=None, delimiter=",").to_numpy().flatten()

y_train= np.array([np.where(i == np.unique(fam[:,0]))[0][0] for i in fam[:,0]])[train_inds]
y_valid= np.array([np.where(i == np.unique(fam[:,0]))[0][0] for i in fam[:,0]])[valid_inds]
y= np.array([np.where(i == np.unique(fam[:,0]))[0][0] for i in fam[:,0]])

y_train= np.array([np.where(i == np.unique(spopped))[0][0] for i in spopped])[train_inds]
y_valid= np.array([np.where(i == np.unique(spopped))[0][0] for i in spopped])[valid_inds]
y= np.array([np.where(i == np.unique(spopped))[0][0] for i in spopped])
color_listx = [
    sns.color_palette("Greys_r", num_colors_per_superpop),
    sns.color_palette("Greens_r", num_colors_per_superpop),
    sns.color_palette("Oranges_r", num_colors_per_superpop),
    # sns.color_palette("PiYG", num_colors_per_superpop * 2),
    sns.color_palette("Blues", num_colors_per_superpop),
    sns.color_palette("PRGn", num_colors_per_superpop * 2),
    # sns.color_palette("BrBG", num_colors_per_superpop),
    sns.color_palette("Reds", num_colors_per_superpop),
    sns.color_palette("YlOrRd", 2 * 3),
    sns.cubehelix_palette(num_colors_per_superpop, reverse=False),
    sns.light_palette("purple", num_colors_per_superpop + 2, reverse=True),
    sns.light_palette("navy", num_colors_per_superpop + 2, reverse=True),
    sns.light_palette("green", num_colors_per_superpop + 2, reverse=True),
    sns.light_palette((210, 90, 60), num_colors_per_superpop + 2, reverse=True, input="husl"),
    sns.light_palette("olive", num_colors_per_superpop + 2, reverse=True),
    sns.light_palette("darkslategray", num_colors_per_superpop + 2, reverse=True),
    sns.light_palette("mediumaquamarine", num_colors_per_superpop + 2, reverse=True),
    sns.dark_palette("indigo", num_colors_per_superpop + 1, reverse=True),
    sns.dark_palette("saddlebrown", num_colors_per_superpop, reverse=True),
    sns.dark_palette("maroon", num_colors_per_superpop + 1, reverse=True),
    sns.dark_palette("gold", num_colors_per_superpop + 1, reverse=True),
    sns.light_palette("lime", num_colors_per_superpop + 1, reverse=True),
    sns.light_palette("orange", num_colors_per_superpop + 1, reverse=True),

]
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

for i in [1]:
    ######################################## Plot settings ###################################
    fsize = 4
    markersize = 10
    lw_scatter_points = 0.1
    lw_figure = 0.01

    #########################################

    sns.set_style(style="whitegrid", rc=None)

    #########################################

    plt.rcParams.update({'xtick.labelsize': 5})
    plt.rcParams.update({'ytick.labelsize': 5})
    plt.rcParams.update({'font.size': 6})
    plt.rcParams.update({'axes.titlesize': 4})
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams.update({'xtick.major.size': 2})
    plt.rcParams.update({'ytick.major.size': 2})
    plt.rcParams.update({'xtick.major.width': 0.0})
    plt.rcParams.update({'ytick.major.width': 0.0})
    plt.rcParams.update({'xtick.major.pad': 0.0})
    plt.rcParams.update({'ytick.major.pad': 0.0})
    plt.rcParams.update({'axes.labelpad': 1.5})
    plt.rcParams.update({'axes.titlepad': 0})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'lines.linewidth': 0.5})
    plt.rcParams.update({'grid.linewidth': 0.1})
    plt.rcParams.update({"scatter.edgecolors": "black"})
    #########################################################################################


def read_h5(filename, dataname):
    '''
    Read data from a h5 file.

    :param filename: directory and filename (with .h5 extension) of file to read from
    :param dataname: name of the datataset in the h5
    :return the data
    '''
    with h5py.File(filename, 'r') as hf:
        data = hf[dataname][:]
        #print(hf.keys())
    return  data
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
    #color_list = np.concatenate(color_listx)
    #color_list = [rgb2hex(color_list[i]) for i in range(len(color_list))]


    edge_list = ["black", "red", "white"]
    t2list = np.unique(quant)
    size_list = [5, 3]
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
                            size=10,
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
        fig.add_trace( go.Line([borderx, bordery], alpha=0.5))

    fig.write_html(f"temp.html")
    fig.show()
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
def latrot(X,theta):
    #rotmat = np.array([[1,0,0],[0,np.cos(theta), -np.sin(theta)],[0,np.sin(theta), np.cos(theta)]])
    rotmat = np.array([[np.cos(theta),0,np.sin(theta)],[0,1, 0],[-np.sin(theta),0,  np.cos(theta)]])
    return (rotmat@X.T).T
def find_angle(coords_M1, quant, k = 20, which = "best", n_iters = 10):

    score = []
    score2 = []

    score.append(knn_f1(quant, coords_M1, [k], return_matches=False)[-1])
    for count, lat_pet in enumerate(np.linspace(0, 2*np.pi, n_iters)):
        for perturbation in np.linspace(0, 2 * np.pi, n_iters):
            if count%2 == 0:
                perturbation = 2*np.pi- perturbation
            coords_M1_go = latrot(coords_M1, lat_pet)
            lat, long = latlong(coords_M1_go[:, 0], coords_M1_go[:, 1], coords_M1_go[:, 2])
            x, y = EqualEarth(lat, long + perturbation)
            encoded_train_mapped = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)

            score2.append(knn_f1(quant, encoded_train_mapped,  [k], return_matches = False)[-1])
            if which =="best":
                bool  = score2[-1] >= np.max(score2)
            else:
                bool  = score2[-1] <= np.min(score2)
            if bool:
                print(f"new best: {knn_f1(quant, encoded_train_mapped, [k])[0]}")
                best_combo = (lat_pet, perturbation)
    plt.figure(figsize=(4,4))
    plt.plot(np.ones_like(score2)*np.mean(score), label = "Score on 3D sphere")
    plt.plot(score2, label = "Score on projection")
    plt.plot(np.ones_like(score2)*np.mean(score2), '--r', label = "Mean projection score")
    plt.ylabel("Classification accuracy")
    plt.xlabel("Perturbation step")
    plt.legend()
    plt.tight_layout()

    plt.savefig("purt.pdf")
    plt.show()

    coords_M1_go = latrot(coords_M1, best_combo[0])
    lat, long = latlong(coords_M1_go[:, 0], coords_M1_go[:, 1], coords_M1_go[:, 2])
    x, y = EqualEarth(lat, long + best_combo[1])
    encoded_train_mapped = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)
    return encoded_train_mapped
def scatter_92(coords, nine,tind = None,vind = None, createfig =True):

    if createfig:
        plt.figure(figsize = (4,4))
    s = 7.5
    lw = 0.2
    plt.scatter(coords[tind,0], coords[tind,1],s = s, label = "Reference/training data", linewidths=lw)  # s is a size of marker
    #plt.colorbar()
    if vind is not None:
        l = len(vind)
        col = ["r", "g", "pink", "yellow"]
        for i in range(4):

            plt.scatter(coords[vind[i*l//4:(i+1)*l//4],0], coords[vind[i*l//4:(i+1)*l//4],1], s =s,marker = "o",c = col[i], label = f"SNP-chip {i+1}", linewidths=lw)  # s is a size of marker
    #plt.legend()
    if createfig:
        plt.show()
def plotly_plot_glob(quant,coordinates,tind=None, vind=None, knn_fails=False,k = 3, top2 = False):

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


    edge_list = ["black", "red", "white"]
    size_list = [8, 3]
    symbol_list = ["circle", "x"]
    if top2:
        wins = knn_f1_top2(quant, coordinates, [k], return_matches=True)
    else:
        wins = knn_f1(quant, coordinates, [k], return_matches=True)
    if tind is not None:
        ind_list = [tind, vind]
    elif knn_fails:
        ind_list = [np.where(wins[-1])[0], np.where(~wins[-1])[0]]

    else:
        ind_list = [np.arange(len(quant))]
    fig = go.Figure()
    for count2, inds in enumerate(ind_list):
        coords = coordinates[inds, :]
        for count, i in enumerate(np.unique(quant)):
            idd = np.where(quant[inds] == i)[0]
            cc = color_list[count % len(color_list)][0]
            'rgba(135, 206, 250, 0.5)'
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
                            size=10,

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
    u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
    r = 0.98
    x = r*np.cos(u) * np.sin(v)
    y = r*np.sin(u) * np.sin(v)
    z = r*np.cos(v)
    cc = np.repeat(np.array('rgb(0.0, 0.0, 0.0)'), len(x))
    cc = np.ones_like(x)
    colorscale = [[0, "rgb(1.0, 1.0, 1.0)"],
                  [1, "rgb(1.0, 1.0, 1.0)"]]
    cmap = plt.get_cmap("tab10")
    #colorscale = [[0, 'rgb' + str(cmap(1)[0:3])],[1, 'rgb' + str(cmap(2)[0:3])]]
    fig.add_trace(
        go.Surface(z = z, x = x, y = y, surfacecolor= cc, opacity = 0.5, colorscale = colorscale,  cmin=0,cmax=1,showscale=False)
    )
    #lines = []
    line_marker = dict(color='#000000', width=2)
    d = 10
    for i, j, k in zip(x[::d,::1], y[::d,::1], z[::d,::1]):
        fig.add_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker,showlegend=False))
    for i, j, k in zip(x[::1,::d].T, y[::1,::d].T, z[::1,::d].T):
        fig.add_trace(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker,showlegend=False))
    fig.write_html(f"test_glob.html")
    fig.show()
def plot_by_thing_kaus(fam, coordinates, proj_fam_file, spop_file, plot_border = False, axisequal = False,createfig=True, marker_size = None):  # Faster, but plotting by label, may misconstrue.


    def get_pop_superpop_list(file):
        '''
        Get a list mapping populations to superpopulations from file.

        :param file: directory, filename and extension of a file mapping populations to superpopulations.
        :return: a (n_pops) x 2 list

        Assumes file contains one population and superpopulation per line, separated by ","  e.g.

        Kyrgyz,Central/South Asia
        Khomani,Sub-Saharan Africa

        '''

        pop_superpop_list = np.genfromtxt(file, usecols=(0, 1), dtype=str, delimiter=",")
        return pop_superpop_list

    # From discussion in https://github.com/matplotlib/matplotlib/issues/11155
    def mscatter(x, y, ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax = plt.gca()
        sc = ax.scatter(x, y,**kw)
        if (m is not None) and (len(m) == len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                    marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc
    def mscatter3(x, y,z, ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax = plt.gca()
        sc = ax.scatter(x, y,z, **kw)
        if (m is not None) and (len(m) == len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                    marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc
    #for i in [1]:
    ######################################## Plot settings ###################################
    fsize = 4
    markersize = 25
    lw_scatter_points = 0.1
    lw_figure = 0.01

    #########################################

    sns.set_style(style="whitegrid", rc=None)

    #########################################

    plt.rcParams.update({'xtick.labelsize': 5})
    plt.rcParams.update({'ytick.labelsize': 5})
    plt.rcParams.update({'font.size': 6})
    plt.rcParams.update({'axes.titlesize': 4})
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams.update({'xtick.major.size': 2})
    plt.rcParams.update({'ytick.major.size': 2})
    plt.rcParams.update({'xtick.major.width': 0.0})
    plt.rcParams.update({'ytick.major.width': 0.0})
    plt.rcParams.update({'xtick.major.pad': 0.0})
    plt.rcParams.update({'ytick.major.pad': 0.0})
    plt.rcParams.update({'axes.labelpad': 1.5})
    plt.rcParams.update({'axes.titlepad': 0})
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'lines.linewidth': 0.5})
    plt.rcParams.update({'grid.linewidth': 0.1})
    plt.rcParams.update({"scatter.edgecolors" : "black"})
    #########################################################################################

    def get_ind_pop_list(filestart):
        '''
        Get a list of individuals and their populations from a .fam file.
        or if that does not exist, tried to find a a .ind file

        :param filestart: directory and file prefix of file containing sample info
        :return: an (n_samples)x(2) list where ind_pop_list[n] = [individial ID, population ID] of the n:th individual
        '''
        try:
            ind_pop_list = np.genfromtxt(filestart + ".fam", usecols=(1, 0), dtype=str)
            # print("Reading ind pop list from " + filestart + ".fam")
        except:
            ind_pop_list = np.genfromtxt(filestart + ".ind", usecols=(0, 2), dtype=str)
            # print("Reading ind pop list from " + filestart + ".ind")

            # probably not general solution
            if ":" in ind_pop_list[0][0]:
                nlist = []
                for v in ind_pop_list:
                    v = v[0]
                    nlist.append(v.split(":")[::-1])
                ind_pop_list = np.array(nlist)

        try:
            # Temporary bad fix. Sometimes fam files are delimited by tabs, if not, it's usually just spaces.
            ind_pop_list = pd.read_csv(filestart + ".fam", header=None).to_numpy()[:, [1, 0]]
        except:
            ind_pop_list = pd.read_csv(filestart + ".fam", header=None, delimiter=" ").to_numpy()[:, [1, 0]]

        return ind_pop_list
    def get_unique_pop_list(filestart):
        '''
        Get a list of unique populations from a .fam file.

        :param filestart: directory and file prefix of file containing sample info
        :return: an (n_pops) list where n_pops is the number of unique populations (=families) in the file filestart.fam
        '''

        #pop_list = np.unique(np.genfromtxt(filestart + ".fam", usecols=(0), dtype=str))
        pop_list = np.unique(np.genfromtxt(filestart + ".fam", usecols=(1), dtype=str))

        return pop_list
    def get_coords_by_pop(filestart_fam, coords, pop_subset=None, ind_pop_list=None):
        '''
        Get the projected 2D coordinates specified by coords sorted by population.

        :param filestart_fam: directory + filestart of fam file
        :param coords: a (n_samples) x 2 matrix of projected coordinates
        :param pop_subset: list of populations to plot samples from, if None then all are returned
        :param ind_pop_list: if specified, gives the ind and population IDs for the samples of coords. If None: assumed that filestart_fam has the correct info.
        :return: a dict that maps a population name to a list of 2D-coordinates (one pair of coords for every sample in the population)


        Assumes that filestart_fam.fam contains samples in the same order as the coordinates in coords.

        '''

        try:
            new_list = []
            for i in range(len(ind_pop_list)):
                new_list.append([ind_pop_list[i][0].decode('UTF-8'), ind_pop_list[i][1].decode('UTF-8')])
            ind_pop_list = np.array(new_list)
        except:
            pass

        if not len(ind_pop_list) == 0:
            unique_pops = np.unique(ind_pop_list[:, 1])

        else:
            ind_pop_list = get_ind_pop_list(filestart_fam)
            unique_pops = get_unique_pop_list(filestart_fam)

        pop_list = ind_pop_list[:, 1]
        coords_by_pop = {}
        for p in unique_pops:
            coords_by_pop[p] = []

        for s in range(len(coords)):
            this_pop = pop_list[s]
            this_coords = coords[s]
            if pop_subset is None:
                coords_by_pop[this_pop].append(this_coords)
            else:
                if this_pop in pop_subset:
                    coords_by_pop[this_pop].append(this_coords)

        return coords_by_pop

    #proj_fam_file = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam"
    #proj_fam_file = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam"
    num_colors_per_superpop = 6

    def plot_coords_by_superpop(pop_coords_dict, outfileprefix, pop_superpop_file, savefig=True, plot_legend=True, epoch="", box_area=0, title=""):
        '''
        Plot the first two dimensions of the given data in a scatter plot.


        :param pop_coords_dict: the data to plot. dictionary mapping population id to a list of n_dim coordinates, n_dim >= 2
        :param outfileprefix: directory and filename, without extension, to save plot to
        :param savefig: save plot to disk
        :param plot_legend: save legend to disk

        '''
        this_pops = list(pop_coords_dict.keys())
        pop_superpop_list = get_pop_superpop_list(pop_superpop_file)
        # only keep the populations that actually appear in the data to plot
        pop_superpop_list = pop_superpop_list[np.isin(pop_superpop_list[:, 0], this_pops)]

        superpops = np.unique(pop_superpop_list[:, 1])
        superpop_dict = {}
        for spop in superpops:
            superpop_dict[spop] = []

        for i in range(len(pop_superpop_list)):
            superpop_dict[pop_superpop_list[i][1]].append(pop_superpop_list[i][0])

        num_colors_per_superpop = 6

        edge_list = ["black" , "white","red"] # ,"red", "white"]
        shape_list = ["o", "v", "<", "s", "p", "H", ">", "p", "D", "X", "*", "d", "h"]
        #shape_list = ["o"] #  "v", "<", "s", "p", "H", ">", "p", "D", "X", "*", "d", "h"]

        sns.set_style(style="white", rc=None)
        color_dict = {}
        max_num_pops = max([len(superpop_dict[spop]) for spop in superpops])
        size = 60

        ################### Plotting the legend #############################
        legends = []

        width = 50.0

        max_pops_per_col = max_num_pops

        fig, axes = plt.subplots(figsize=(6.0 / 2.0, 1.5 * max_num_pops / 4.0))
        plt.setp(axes.spines.values(), linewidth=0.0)

        row = 0
        col = 0.0
        num_legend_entries = 0

        for spop in range(len(superpops)):

            if spop > 0:
                row += 1

            spopname = superpops[spop]
            this_pops = superpop_dict[superpops[spop]]

            # counter of how many times same color is used: second time want to flip the shapes
            time_used = spop // len(color_list)

            this_pop_color_list = list(map(rgb2hex, color_list[spop % len(color_list)][0:num_colors_per_superpop]))

            if time_used == 0:
                combos = np.array(np.meshgrid(shape_list, this_pop_color_list, edge_list)).T.reshape(-1, 3)
            else:
                combos = np.array(np.meshgrid((shape_list[::-1]), this_pop_color_list, edge_list)).T.reshape(-1, 3)

            this_superpop_points = []
            for p in range(len(this_pops)):
                assert not this_pops[p] in color_dict.keys()
                color_dict[this_pops[p]] = combos[p]
                point = plt.scatter([-1], [-1], color=combos[p][1], marker=combos[p][0], s=size, edgecolors=combos[p][2],
                                    label=this_pops[p])
                this_superpop_points.append(point)

            # if we swith to next column
            if num_legend_entries + len(this_pops) > max_pops_per_col:
                col += 1
                row = 0
                num_legend_entries = 0

            l = plt.legend(this_superpop_points,
                           [p for p in this_pops],
                           title=r'$\bf{' + superpops[spop] + '}$',
                           # x0, y0, width, height
                           bbox_to_anchor=(float(col),
                                           1 - (float(num_legend_entries + row) / max_pops_per_col),
                                           0,
                                           0),
                           loc='upper left',
                           markerscale=2.5,
                           fontsize=12)

            l.get_title().set_fontsize('14')
            num_legend_entries += len(this_pops) + 1
            legends.append(l)

        for l in legends:
            axes.add_artist(l)

        plt.xlim(left=2, right=width)
        plt.ylim(bottom=0, top=width)
        plt.xticks([])
        plt.yticks([])

        if plot_legend:
            plt.savefig("{0}_legend.pdf".format(outfileprefix), bbox_inches="tight")
        plt.close()

        ################### Plotting the samples #############################

        sns.set_style(style="whitegrid", rc=None)
        if createfig:
            fig = plt.figure(figsize=(fsize, fsize), linewidth=lw_figure)
        if coordinates.shape[-1] == 3:
           ax = fig.add_subplot(projection='3d')

        scatter_points = []
        colors = []
        markers = []
        edgecolors = []
        box_color = '--k'
        if not box_area == 0:
            box = np.sqrt(box_area)
            plot_box = True
        else:
            box = 0
            plot_box = False
        for spop in range(len(superpops)):
            #spopname = superpops[spop]
            #if spopname == "aMex":
            #    markersize = 25
            #else:
            #    markersize = 75

            this_pops = superpop_dict[superpops[spop]]

            for pop in this_pops:
                if pop in pop_coords_dict.keys():

                    this_fam_coords = np.array(pop_coords_dict[pop])
                    # print(np.max(np.abs(this_fam_coords)))
                    try:
                        if np.max(np.abs(this_fam_coords)) > box:
                            box_color = '--r'
                    except:
                        pass
                    if len(this_fam_coords) > 0:
                        scatter_points.extend(this_fam_coords)
                        colors.extend([color_dict[pop][1] for i in range(len(this_fam_coords))])
                        markers.extend([color_dict[pop][0] for i in range(len(this_fam_coords))])
                        edgecolors.extend([color_dict[pop][2] for i in range(len(this_fam_coords))])


                else:
                    # print("Population NOT in data: {0}".format(pop))
                    pass
        sp_array = np.array(scatter_points)*1.01
        order = np.arange(len(colors))
        np.random.shuffle(order)
        if coordinates.shape[-1] == 2:

            if marker_size == None:
                markersize = 25
            else:
                markersize = marker_size
            mscatter(sp_array[order, 0],
                     sp_array[order, 1],
                     color=[colors[x] for x in order],
                     m=[markers[x] for x in order],
                     s=markersize,
                     edgecolors="k", #[edgecolors[x] for x in order],
                     label=pop,
                     linewidth=lw_scatter_points)
        else:
            if marker_size==None:
                markersize = 10
            else:
                markersize =marker_size
            mscatter3(sp_array[order, 0],
                     sp_array[order, 1],
                     sp_array[order, 2],
                     color=[colors[x] for x in order],
                     m=[markers[x] for x in order],
                     s=markersize,
                     edgecolors=[edgecolors[x] for x in order],
                     label=pop,
                     linewidth=lw_scatter_points,
                      ax = ax)

            u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
            r = 1
            x = r * np.cos(u) * np.sin(v)
            y = r * np.sin(u) * np.sin(v)
            z = r * np.cos(v)
            from matplotlib import cm

            #ax.plot_wireframe(x,y,z, rstride=10, cstride=10, color = "black")
            #ax.plot_surface(x,y,z, color = "blue", shade = False, alpha =1)
            surf = ax.plot_surface(x,y,z, color = "white", alpha =0.9,linewidth=0.0, shade = False)
            #surf = ax.plot_surface(x,y,z, color = "blue", shade = False, alpha =1)
            surf.set_edgecolor((0, 0, 0, 0))

        if axisequal:
            plt.axis("equal")
        plot_box = False
        if plot_box:
            if box_color == '--r':
                alp = 1
            else:
                alp = 0.5
            plt.plot([-box, box], [-box, -box], box_color, alpha=alp)
            plt.plot([-box, box], [box, box], box_color, alpha=alp)
            plt.plot([-box, -box], [-box, box], box_color, alpha=alp)
            plt.plot([box, box], [box, -box], box_color, alpha=alp)

            plt.xlim([-box * 1.2, box * 1.2])
            plt.ylim([-box * 1.2, box * 1.2])
        else:
            # plt.xlim([-1.2,1.2])
            # plt.ylim([-1.2, 1.2])
            pass
        plt.title(title)
        #if savefig:
        #    plt.savefig(outfileprefix + ".pdf")
        #    print("saving plot to " + outfileprefix + ".pdf")
        #plt.close()

        if plot_border:
            # draw sphere
            u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:200j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)

            latlong(np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1))
            lat, long = latlong(np.reshape(x, -1), np.reshape(y, -1), np.reshape(z, -1))
            x, y = EqualEarth(lat, long)
            shape = alphashape.alphashape(list(zip(x, y)), alpha=2)
            borderx, bordery = shape.exterior.coords.xy
            plt.plot(borderx, bordery, alpha = 0.5)
        return scatter_points, colors, markers, edgecolors
    if spop_file== "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations":
        color_list = [
            sns.color_palette("Greys_r", num_colors_per_superpop),
            sns.color_palette("Greens_r", num_colors_per_superpop),
            sns.color_palette("Oranges_r", num_colors_per_superpop),
            # sns.color_palette("PiYG", num_colors_per_superpop * 2),
            sns.color_palette("Blues", num_colors_per_superpop),
            sns.color_palette("PRGn", num_colors_per_superpop*2),
            # sns.color_palette("BrBG", num_colors_per_superpop),
            sns.color_palette("Reds", num_colors_per_superpop),
            sns.color_palette("YlOrRd", 2*3),
            sns.cubehelix_palette(num_colors_per_superpop, reverse=False),
            sns.light_palette("purple", num_colors_per_superpop+2, reverse=True),
            sns.light_palette("navy", num_colors_per_superpop+2, reverse=True),
            sns.light_palette("green", num_colors_per_superpop+2, reverse=True),
            sns.light_palette((210, 90, 60), num_colors_per_superpop+2, reverse=True, input="husl"),
            sns.light_palette("olive", num_colors_per_superpop+2, reverse=True),
            sns.light_palette("darkslategray", num_colors_per_superpop+2, reverse=True),
            sns.light_palette("mediumaquamarine", num_colors_per_superpop+2, reverse=True),
            sns.dark_palette("indigo", num_colors_per_superpop+1, reverse=True),
            sns.dark_palette("saddlebrown", num_colors_per_superpop+1, reverse=True),
            sns.dark_palette("maroon", num_colors_per_superpop+1, reverse=True),
            sns.dark_palette("gold", num_colors_per_superpop+1, reverse=True),
            sns.light_palette("lime", num_colors_per_superpop+1, reverse=True),
            sns.light_palette("orange", num_colors_per_superpop+1, reverse=True),

        ]
    else:

        color_list = [
        sns.color_palette("Greys_r", num_colors_per_superpop),
        sns.color_palette("Greens_r", num_colors_per_superpop),
        sns.color_palette("Oranges_r", num_colors_per_superpop),
        sns.color_palette("PiYG", num_colors_per_superpop * 2),
        sns.color_palette("Blues", num_colors_per_superpop),
        sns.color_palette("PRGn", num_colors_per_superpop * 2),
        sns.color_palette("BrBG", num_colors_per_superpop),
        sns.color_palette("Reds", num_colors_per_superpop),
        sns.color_palette("YlOrRd", 2 * 3),
        sns.cubehelix_palette(num_colors_per_superpop, reverse=False),
        sns.light_palette("purple", num_colors_per_superpop + 2, reverse=True),
        sns.light_palette("navy", num_colors_per_superpop + 2, reverse=True),
        sns.light_palette("green", num_colors_per_superpop + 2, reverse=True),
        sns.light_palette((210, 90, 60), num_colors_per_superpop + 2, reverse=True, input="husl"),
        sns.light_palette("olive", num_colors_per_superpop + 2, reverse=True),
        sns.light_palette("darkslategray", num_colors_per_superpop + 2, reverse=True),
        sns.light_palette("mediumaquamarine", num_colors_per_superpop + 2, reverse=True),
        sns.dark_palette("indigo", num_colors_per_superpop + 1, reverse=True),
        sns.dark_palette("saddlebrown", num_colors_per_superpop, reverse=True),
        sns.dark_palette("maroon", num_colors_per_superpop + 1, reverse=True),
        sns.dark_palette("gold", num_colors_per_superpop + 1, reverse=True),
        sns.light_palette("lime", num_colors_per_superpop + 1, reverse=True),
        sns.light_palette("orange", num_colors_per_superpop + 1, reverse=True),

    ]


    #spop_file = pop_superpop_file = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations"
    #spop_file = pop_superpop_file = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations"
    coords_by_pop = get_coords_by_pop(proj_fam_file, coordinates, ind_pop_list=np.vstack([fam[:, 1], fam[:, 0]]).T)
    plot_coords_by_superpop(coords_by_pop, "",pop_superpop_file = spop_file, title= "")
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
    #color_list = np.concatenate(color_listx)
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
        name = test3
    fig.write_html(f"{name}.html")
    fig.show()




# Dog data

coords_M1 = read_h5("encoded_data.h5", f"{5000}_encoded_train")
coords_M1_triplet = read_h5("encoded_data_triplet.h5", f"{5000}_encoded_train")

proj_fam_file_dog = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam"
spop_file_dog  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations"

encoded_train_mapped = find_angle(coords_M1,fam[:,0],k = 20) # k = 20 used in paper
encoded_train_mapped_triplet = find_angle(coords_M1_triplet, fam[:,0],k = 20)

geno_data = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_filtered.parquet").to_numpy()
mcg = scipy.stats.mode(geno_data,axis = 1)
idx = np.where(geno_data == 9 )
geno_data[idx] = mcg.mode[idx[0]]

geno_data_normed  = (geno_data - np.mean(geno_data, axis = 1, keepdims = True)) / np.std(geno_data, axis = 1, keepdims = True)
X_embedded = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(geno_data_normed.T)
X_embedded3d = TSNE(n_components=3, learning_rate='auto',init='random').fit_transform(geno_data_normed.T)
X_embedded2 = TSNE(n_components=2, learning_rate='auto',perplexity = 1000, init='random').fit_transform(geno_data_normed.T)

pca = PCA(n_components=2)
pca.fit(geno_data_normed[:,train_inds].T)
X_PCA = pca.transform(geno_data_normed.T)

pca3 = PCA(n_components=3)
pca3.fit(geno_data_normed[:,train_inds].T)
X_PCA3 = pca3.transform(geno_data_normed.T)


# TEST t-SNE 3d PLOT
tsne3normed = X_embedded3d / np.sqrt(np.sum(X_embedded3d**2, axis = 1, keepdims=True))
pca3normed = X_PCA3 / np.sqrt(np.sum(X_PCA3**2, axis = 1, keepdims=True))

plotly_plot(spops, X_embedded3d, train_inds, valid_inds)
plotly_plot(fam[:,0], X_embedded3d, train_inds, valid_inds)


plotly_plot(spops, X_PCA3, train_inds, valid_inds)
plotly_plot(spops, X_PCA, train_inds, valid_inds)
tsne3_mapped = find_angle(tsne3normed, fam[:,0])
pca3_mapped = find_angle(pca3normed, fam[:,0])


# FIGURE S3
plt.figure()
plot_by_thing_kaus(fam,tsne3_mapped,proj_fam_file_dog, spop_file_dog,axisequal=False, plot_border = True)
plt.title("t-SNE",fontsize=10)
plt.tight_layout()
plt.savefig("tsne3mapped2.pdf")
plt.figure()
plot_by_thing_kaus(fam,pca3_mapped,proj_fam_file_dog, spop_file_dog,axisequal=False, plot_border = True)
plt.title("PCA",fontsize=10)
plt.tight_layout()
plt.savefig("pca3mapped2.pdf")


# VALUES IN TABLE S1
knn_f1_tt(fam[:,0], tsne3_mapped, train_inds, valid_inds, [3])
knn_f1_tt(spops, tsne3_mapped, train_inds, valid_inds, [3])

knn_f1_tt(fam[:,0], pca3_mapped, train_inds, valid_inds,[3])
knn_f1_tt(spops, pca3_mapped, train_inds, valid_inds,[3])

knn_f1_tt(fam[:,0], X_PCA, train_inds, valid_inds,[3])
knn_f1_tt(spops, X_PCA, train_inds, valid_inds,[3])

knn_f1_tt(fam[:,0], encoded_train_mapped, train_inds, valid_inds,[3])
knn_f1_tt(spops, encoded_train_mapped, train_inds, valid_inds,[3])

knn_f1_tt(fam[:,0], X_embedded, train_inds, valid_inds,[3])
knn_f1_tt(spops, X_embedded, train_inds, valid_inds,[3])


# FIGURE  6
fig = plt.figure(figsize=(2*fsize, 2*fsize), linewidth=lw_figure)
plt.subplot(2,2,1)
plot_by_thing_kaus(fam, X_PCA, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("PCA",fontsize=10)
plt.subplot(2,2,2)
plot_by_thing_kaus(fam, X_embedded, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("t-SNE",fontsize=10)
plt.subplot(2,2,3)
plot_by_thing_kaus(fam, encoded_train_mapped_triplet, proj_fam_file_dog, spop_file_dog, plot_border= True, axisequal=False,createfig=False)
plt.title("Triplet",fontsize=10)
plt.subplot(2,2,4)
plot_by_thing_kaus(fam, encoded_train_mapped, proj_fam_file_dog, spop_file_dog, plot_border= True, axisequal=False,createfig=False)
plt.title("Centroid",fontsize=10)
plt.tight_layout()
plt.savefig("dog_centroid_testplt222.pdf")



# Preparation of values in table 1

method = [coords_M1, encoded_train_mapped, encoded_train_mapped_triplet, X_embedded, X_PCA, coords_M1_triplet]
method_name = ["globe", "centroid", "triplet", "tsne", "PCA", "globe_triplet"]
quant = [spops, fam[:,0]]
quant_name = ["Superpopulation" , "Subpopulation"]
dir_dog = {}
for qn,q in enumerate(quant):
    for mn, m in enumerate(method):
        print(f"\n {quant_name[qn]}, {method_name[mn]} ")
        dir_dog[quant_name[qn] + method_name[mn]]  =knn_f1_tt(q, m, train_inds, valid_inds , [3])



# Human data + evaluation

geno_data_valmiss = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered_valmiss.parquet").to_numpy()
geno_data = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.parquet").to_numpy()
nine = np.sum(geno_data_valmiss == 9, axis = 0) / geno_data_valmiss.shape[0]

proj_fam_file = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam"
spop_file  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations"

coords_M1_human = read_h5("encoded_datah_full.h5", f"{5000}_encoded_train")
coords_M1_valmiss = read_h5("encoded_data_valmiss.h5", f"{5000}_encoded_train")

encoded_train_mapped_human = find_angle(coords_M1_human,fam_human[:,0], k = 20)
encoded_train_mapped_human_valmiss = find_angle(coords_M1_valmiss,spops_human,k = 20)

l = len(valid_inds_human)
missing_label = np.zeros(len(coords_M1_human[:,0]))
missing_label[train_inds] = 0
for i in range(4):
    missing_label[valid_inds_human[i*l//4 :(i+1)*l//4]] = i+1

# set missing to most common
mcg = scipy.stats.mode(geno_data,axis = 1)
idx = np.where(geno_data == 9 )
geno_data[idx] = mcg.mode[idx[0]]#[:,0]
mcg = scipy.stats.mode(geno_data_valmiss,axis = 1)
idx = np.where(geno_data_valmiss == 9 )
geno_data_valmiss[idx] = mcg.mode[idx[0]]#[:,0]

geno_data_normed = (geno_data - np.mean(geno_data, axis = 1, keepdims = True)) / np.std(geno_data, axis = 1, keepdims = True)
X_embedded_human = TSNE(n_components=2, learning_rate="auto",init='random').fit_transform(geno_data.T)
pca = PCA(n_components=2)
pca.fit(geno_data_normed[:,train_inds_human].T)
X_PCA_human = pca.transform(geno_data_normed.T)

geno_data_normed_valmiss = (geno_data_valmiss - np.mean(geno_data_valmiss, axis = 1, keepdims = True)) / np.std(geno_data_valmiss, axis = 1, keepdims = True)
X_embedded_human_valmiss = TSNE(n_components=2, learning_rate=200, perplexity = 10, init='random').fit_transform(geno_data_normed_valmiss.T)
X_embedded_human_valmiss = TSNE(n_components=2, learning_rate="auto",init='random').fit_transform(geno_data_normed_valmiss.T)
pca = PCA(n_components=2)
pca.fit(geno_data_normed_valmiss[:,train_inds_human].T)
X_PCA_human_valmiss = pca.transform(geno_data_normed_valmiss.T)


# FIGURE  8
fig = plt.figure(figsize=(2*fsize, 2*fsize), linewidth=lw_figure)
plt.subplot(2,2,1)
plot_by_thing_kaus(fam_human, encoded_train_mapped_human_valmiss, proj_fam_file, spop_file, plot_border= False, axisequal=False,createfig=False,marker_size = 10)
plt.title("Centroid",fontsize=10)
plt.subplot(2,2,2)
plot_by_thing_kaus(fam_human, X_embedded_human_valmiss, proj_fam_file, spop_file, plot_border= False, axisequal=False,createfig=False,marker_size = 10)
plt.title("t-SNE",fontsize=10)
plt.subplot(2,2,3)
scatter_92(encoded_train_mapped_human_valmiss, nine, train_inds_human, valid_inds_human, createfig=False)
plt.subplot(2,2,4)
scatter_92(X_embedded_human_valmiss, nine, train_inds_human, valid_inds_human, createfig=False)
plt.tight_layout()
plt.savefig("human_bigfig22.pdf")



# init for human and valmiss tables.
method = [encoded_train_mapped_human, X_embedded_human, X_PCA_human, coords_M1_human]
method_name = ["centroid", "tsne", "PCA", "centroid_globe"]
quant = [spops_human, fam_human[:,0]]
quant_name = ["Superpopulation" , "Subpopulation"]
dir_human = {}
for qn,q in enumerate(quant):
    for mn, m in enumerate(method):
        print(f"\n {quant_name[qn]}, {method_name[mn]} ")
        dir_human[quant_name[qn] + method_name[mn]]  =knn_f1_tt(q, m, train_inds_human, valid_inds_human , [3])

method = [encoded_train_mapped_human_valmiss, X_embedded_human_valmiss, X_PCA_human_valmiss, coords_M1_valmiss]
method_name = ["centroid", "tsne", "PCA", "centroid_globe"]
quant = [spops_human, fam_human[:,0]]
quant_name = ["Superpopulation" , "Subpopulation"]
dir_human_valmiss = {}
for qn,q in enumerate(quant):
    for mn, m in enumerate(method):
        print(f"\n {quant_name[qn]}, {method_name[mn]} ")
        dir_human_valmiss[quant_name[qn] + method_name[mn]]  =knn_f1_tt(q, m, train_inds_human, valid_inds_human , [3])


# TABLE 1 Latex format
print( "{\\bf Method} & {\\bf Dataset} & {\\bf Subpop acc.} & validation & {\\bf Superpop acc. }& validation \\\\")

print("\\hline")

print(f" PCA & Dog &  {dir_dog['SubpopulationPCA'][-1]:.4f} & {dir_dog['SubpopulationPCA'][1]:.4f} &  {dir_dog['SuperpopulationPCA'][-1]:.4f} & {dir_dog['SuperpopulationPCA'][1]:.4f} \\\\")
print(f" t-SNE & Dog & {dir_dog['Subpopulationtsne'][-1]:.4f} &  {dir_dog['Subpopulationtsne'][1]:.4f}  & {dir_dog['Superpopulationtsne'][-1]:.4f} &   {dir_dog['Superpopulationtsne'][1]:.4f} \\\\" )
print(f" Triplet & Dog & {dir_dog['Subpopulationglobe_triplet'][-1]:.4f} &  {dir_dog['Subpopulationglobe_triplet'][1]:.4f} &  {dir_dog['Superpopulationglobe_triplet'][-1]:.4f} & {dir_dog['Superpopulationglobe_triplet'][1]:.4f} \\\\")
print(f" Centroid & Dog & {dir_dog['Subpopulationglobe'][-1]:.4f} &  {dir_dog['Subpopulationglobe'][1]:.4f} &  {dir_dog['Superpopulationglobe'][-1]:.4f} & {dir_dog['Superpopulationglobe'][1]:.4f} \\\\")
print("\\hline")

print(f" PCA & Human & {dir_human['SubpopulationPCA'][-1]:.4f} &  {dir_human['SubpopulationPCA'][1]:.4f} & {dir_human['SuperpopulationPCA'][-1]:.4f} & {dir_human['SuperpopulationPCA'][1]:.4f} \\\\")
print(f" t-SNE & Human & {dir_human['Subpopulationtsne'][-1]:.4f} &  {dir_human['Subpopulationtsne'][1]:.4f} &  {dir_human['Superpopulationtsne'][-1]:.4f} & {dir_human['Superpopulationtsne'][1]:.4f}  \\\\")
print(f" Centroid & Human & {dir_human['Subpopulationcentroid_globe'][-1]:.4f} &  {dir_human['Subpopulationcentroid_globe'][1]:.4f} & {dir_human['Superpopulationcentroid_globe'][-1]:.4f} & {dir_human['Superpopulationcentroid_globe'][1]:.4f} \\\\")
print("\\hline")

print(f" PCA & Hu. masked & {dir_human_valmiss['SubpopulationPCA'][-1]:.4f} &  {dir_human_valmiss['SubpopulationPCA'][1]:.4f} &  {dir_human_valmiss['SuperpopulationPCA'][-1]:.4f}  & {dir_human_valmiss['SuperpopulationPCA'][1]:.4f} \\\\")
print(f" t-SNE &  Hu. masked & {dir_human_valmiss['Subpopulationtsne'][-1]:.4f} & {dir_human_valmiss['Subpopulationtsne'][1]:.4f} &  {dir_human_valmiss['Superpopulationtsne'][-1]:.4f}  & {dir_human_valmiss['Superpopulationtsne'][1]:.4f}  \\\\")
print(f" Centroid &  Hu. masked & {dir_human_valmiss['Subpopulationcentroid_globe'][-1]:.4f} &  {dir_human_valmiss['Subpopulationcentroid_globe'][1]:.4f} & {dir_human_valmiss['Superpopulationcentroid_globe'][-1]:.4f} & {dir_human_valmiss['Superpopulationcentroid_globe'][1]:.4f} \\\\")




# FIGURE 7
K = 250
t1s = knn_f1(spops, X_embedded,np.arange(1,K))[1]
c1s= knn_f1(spops, coords_M1,np.arange(1,K))[1]
p1s= knn_f1(spops, X_PCA,np.arange(1,K))[1]

plt.figure(figsize = (6,3))
plt.plot(t1s,label = "2D t-SNE")
plt.plot(c1s,label = "Centroid")
plt.plot(p1s,label = "2D PCA")
plt.legend()

plt.xlabel("Number of neighbors K")
plt.ylabel("KNN population classification score")
plt.savefig("test_largek_dawg22.pdf")
plt.show()


geno_data_valmiss = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered_valmiss.parquet").to_numpy()
geno_data_valmiss_normed = (geno_data_valmiss - np.mean(geno_data_valmiss, axis = 1, keepdims = True)) / np.std(geno_data_valmiss, axis = 1, keepdims = True)

X_embedded_valmiss9 = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(geno_data_valmiss_normed.T)

idx = np.where(geno_data_valmiss == 9 )
geno_data_valmiss[idx] = -1
geno_data_valmiss_normed = (geno_data_valmiss - np.mean(geno_data_valmiss, axis = 1, keepdims = True)) / np.std(geno_data_valmiss, axis = 1, keepdims = True)

X_embedded_valmiss1 = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(geno_data_valmiss_normed.T)

geno_data_valmiss = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered_valmiss.parquet").to_numpy()
idx = np.where(geno_data_valmiss == 9 )
mcg = scipy.stats.mode(geno_data_valmiss,axis = 1)
geno_data_valmiss[idx] = mcg.mode[idx[0]]
geno_data_valmiss_normed = (geno_data_valmiss - np.mean(geno_data_valmiss, axis = 1, keepdims = True)) / np.std(geno_data_valmiss, axis = 1, keepdims = True)

X_embedded_valmissmcg = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(geno_data_valmiss_normed.T)


plt.figure(figsize = (8,12))
plt.subplot(3,2,1)
plot_by_thing_kaus(fam_human, X_embedded_valmiss9, proj_fam_file, spop_file, createfig=False, plot_border = False)
plt.title("Missing = 9", fontsize = 10)
plt.subplot(3,2,2)
scatter_92(X_embedded_valmiss9, nine, train_inds_human, valid_inds_human, createfig=False)
plt.title("Missing = 9", fontsize = 10)

plt.subplot(3,2,3)
plot_by_thing_kaus(fam_human, X_embedded_valmiss1, proj_fam_file, spop_file, createfig=False, plot_border = False)
plt.title("Missing = -1", fontsize = 10)
plt.subplot(3,2,4)
scatter_92(X_embedded_valmiss1, nine, train_inds_human, valid_inds_human, createfig=False)
plt.title("Missing = -1", fontsize = 10)
plt.subplot(3,2,5)
plot_by_thing_kaus(fam_human, X_embedded_valmissmcg, proj_fam_file, spop_file, createfig=False, plot_border = False)
plt.title("Missing = MCG", fontsize = 10)
plt.subplot(3,2,6)
scatter_92(X_embedded_valmissmcg, nine, train_inds_human, valid_inds_human, createfig=False)
plt.title("Missing = MCG", fontsize = 10)
plt.legend()
plt.tight_layout()
plt.savefig("diff_missing2.pdf")

geno_data_valmiss = pd.read_parquet("/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered_valmiss.parquet").to_numpy()
miss_idx = np.where(np.sum(geno_data_valmiss[:,valid_inds] ==9,axis = 1) == 0)[0]

geno_data_no_miss = geno_data_valmiss[miss_idx,:]

idx = np.where(geno_data_no_miss == 9 )
mcg = scipy.stats.mode(geno_data_no_miss,axis = 1)
geno_data_no_miss[idx] = mcg.mode[idx[0]]
X_embedded_no_valmiss= TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(geno_data_no_miss.T)
plt.figure()
scatter_92(X_embedded_no_valmiss, nine, train_inds_human, valid_inds_human, createfig=False)
plot_by_thing_kaus(fam_human, X_embedded_no_valmiss, proj_fam_file, spop_file, createfig=False, plot_border = False)
plt.show()


knn_f1_tt(spops_human, X_embedded_valmiss9, train_inds_human, valid_inds_human, [3])
knn_f1_tt(spops_human, X_embedded_valmiss1, train_inds_human, valid_inds_human, [3])
knn_f1_tt(spops_human, X_embedded_valmissmcg, train_inds_human, valid_inds_human, [3])
knn_f1_tt(spops_human, coords_M1_valmiss, train_inds_human, valid_inds_human, [3])


knn_f1_tt(fam_human[:,0], X_embedded_valmiss9, train_inds_human, valid_inds_human, [3])
knn_f1_tt(fam_human[:,0], X_embedded_valmiss1, train_inds_human, valid_inds_human, [3])
knn_f1_tt(fam_human[:,0], X_embedded_valmissmcg, train_inds_human, valid_inds_human, [3])
knn_f1_tt(fam_human[:,0], coords_M1_valmiss, train_inds_human, valid_inds_human, [3])



plot_by_thing_kaus(fam_human,X_embedded_valmissmcg,proj_fam_file, spop_file,axisequal=False, plot_border=True, marker_size =10 )
plt.tight_layout()
plt.savefig("test.pdf")

encoded_train_mappedworst = find_angle(coords_M1_valmiss,fam_human[:,0],k = 3, which= "worst")

knn_f1(fam_human[:,0], encoded_train_mappedworst, [3])

plot_by_thing_kaus(fam_human,encoded_train_mappedworst,proj_fam_file, spop_file,axisequal=False, plot_border=True, marker_size =10 )
plt.tight_layout()
plt.savefig("worst2.pdf")

plotly_plot_glob(spops,coords_M1)
plotly_plot(spops,encoded_train_mapped, plot_border = True)


plotly_plot(spops,X_PCA,name = "pca_2d")

