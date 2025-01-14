import numpy as np
import matplotlib.pyplot as plt
import alphashape
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
import h5py
import scipy
import seaborn as sns
from matplotlib.colors import rgb2hex
import pandas as pd
from scipy.spatial import distance_matrix
import pickle

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

# File handles needed for kaus plot.
proj_fam_file_dog = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/All_Pure_150k.fam"
spop_file_dog  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/dog/dog_superpopulations"
proj_fam_file_human = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HumanOrigins2067_filtered.fam"
spop_file_human  = "/home/filip/Documents/ContrastiveLosses/Contrastive_Losses/gcae/Data/HO_superpopulations"



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
def scatter_92(coords, nine,tind = None,vind = None, createfig =True, s = 7.5):

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


    coords_by_pop = get_coords_by_pop(proj_fam_file, coordinates, ind_pop_list=np.vstack([fam[:, 1], fam[:, 0]]).T)
    plot_coords_by_superpop(coords_by_pop, "",pop_superpop_file = spop_file, title= "")

# Evalution metrics
def local_adherence(reference_neighbors, embedding, kmax = 100, step = 1 ):
    # LOCAL - Check K-nearest neighbor in embedding space, and in genotype space. Compare the overlap between these.

    neibs = np.min([reference_neighbors.shape[1], kmax])

    d_embedding = distance_matrix(embedding, embedding)
    nbrs2 = NearestNeighbors(n_neighbors=neibs, algorithm='ball_tree').fit(d_embedding)
    distances2, indices_embedding = nbrs2.kneighbors(d_embedding)

    score_embedding = []

    k_vec = []
    for k in np.arange(3,kmax, step):
        score_embedding_temp = 0

        for i in range(embedding.shape[0]):
            score_embedding_temp += np.intersect1d(reference_neighbors[i, 1:k], indices_embedding[i, 1:k]).shape[0]

        score_embedding.append(score_embedding_temp /(reference_neighbors.shape[0] * len(reference_neighbors[i, 1:k])) )

        k_vec.append(len(reference_neighbors[i, 1:k]))
    return score_embedding, k_vec
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
def local_adherence_fixed(reference_neighbors, embedding, k = 30):
    neibs = k

    d_embedding = distance_matrix(embedding, embedding)
    nbrs2 = NearestNeighbors(n_neighbors=neibs, algorithm='ball_tree').fit(d_embedding)
    distances2, indices_embedding = nbrs2.kneighbors(d_embedding)
    score_temp = []
    for i in range(reference_neighbors.shape[0]):
        score_temp.append(np.intersect1d(reference_neighbors[i, 1:k], indices_embedding[i, 1:k]).shape[0])

    return k- np.array(score_temp), k
def get_eval_metrics(subpops_, spops_, embedding_, reference_neighbors_, train_inds_, valid_inds_, k_vec_, fixed_k_):
    train_knn_acc, valid_knn_acc, total_knn_acc = knn_f1_tt(subpops_, embedding_, train_inds_, valid_inds_, k_vec_, verbose=False)
    glob_spop_umap_accuracy, pop_embedding = global_spop(subpops_, spops_, embedding_)
    local_preservation_score,  local_k =  local_adherence_fixed(reference_neighbors_, embedding_, fixed_k_)
    local_adherence_score, local_adherence_k_vec = local_adherence(reference_neighbors_, embedding_, kmax = 100, step = 1)

    r_dict = {"train_knn_acc":train_knn_acc, "valid_knn_acc":valid_knn_acc,
              "glob_spop_umap_accuracy":glob_spop_umap_accuracy,"local_preservation_score":local_preservation_score, "suppop_embedding":pop_embedding,
              "local_adherence_score":local_adherence_score , "local_adherence_k_vec":local_adherence_k_vec}
    return r_dict
def global_rank_score(reference_distance_matrix, embedding_distance_matrix, pop_to_do):
     # Computes the Kendall's tau rank correlation on the average inter-population distances between two distance matrices.
     # In my case here, I want to compare the rankings based on the Manhattan distance in the genotypes,
     # and the L2 distances in the embeddings.

    nspops = len(np.unique(pop_to_do))
    dct = {}
    for i in np.unique(pop_to_do):
        dct[i] = np.where(pop_to_do == i)[0]

    inter_pop_distance_reference = np.zeros((nspops, nspops))
    inter_pop_distance_embedding = np.zeros((nspops, nspops))

    for counti, i in enumerate(dct):
        for countj, j in enumerate(dct):

            # These are the distances we want to take the average of.
            ax1 = np.repeat(dct[i], len(dct[j]))[:,np.newaxis]
            ax2 = np.tile(dct[j], len(dct[i]))[:,np.newaxis]
            inds = np.concatenate((ax1, ax2), axis = 1)

            # inter-population distances in reference
            distances_reference= reference_distance_matrix[inds[:,0], inds[:,1]]
            inter_pop_distance_reference[counti,countj] = np.mean(distances_reference)
            # inter-population distances in embedding
            distances_embedding= embedding_distance_matrix[inds[:,0], inds[:,1]]
            inter_pop_distance_embedding[counti,countj] = np.mean(distances_embedding)

    nbrs2 = NearestNeighbors(n_neighbors=nspops, algorithm='ball_tree').fit(inter_pop_distance_reference)
    distances2, indices_reference = nbrs2.kneighbors(inter_pop_distance_reference)
    nbrs2 = NearestNeighbors(n_neighbors=nspops, algorithm='ball_tree').fit(inter_pop_distance_embedding)
    distances2, indices_embedding = nbrs2.kneighbors(inter_pop_distance_embedding)

    score_embedding = []
    for nn in range(nspops):
        ranking_reference = np.where(indices_reference[nn, :] == np.arange(nspops)[:, np.newaxis])[1]
        ranking_embedding = np.where(indices_embedding[nn, :] == np.arange(nspops)[:, np.newaxis])[1]
        score_embedding.append(np.array(scipy.stats.kendalltau(ranking_reference, ranking_embedding)[0]))
    return np.array(score_embedding)

nine_labels = np.zeros(len(spops_human))
nine_labels[train_inds_human ] = 0
l = len(valid_inds_human)
for i in range(4):
    nine_labels[valid_inds_human[i * l // 4:(i + 1) * l // 4]] = i+1


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



neibs = 200
# LOCAL - Check K-nearest neighbor in embedding space, and in genotype space. Compare the overlap between these.

dd = np.load("saved_files/distance_dog.npy")
reference_distance = dd
nbrs = NearestNeighbors(n_neighbors=neibs, algorithm='ball_tree').fit(reference_distance)
_, reference_neighbors = nbrs.kneighbors(reference_distance)

reference_distance_human =  np.load("saved_files/distance_human.npy")
nbrs = NearestNeighbors(n_neighbors=neibs, algorithm='ball_tree').fit(reference_distance_human)
_, reference_neighbors_human = nbrs.kneighbors(reference_distance_human)



# Load in Neural network results, contrastive and popvae for both human and dog :
# dog:
coords_contrastive = read_h5("saved_files/centroid_dog.h5", f"{5000}_encoded_train")
encoded_train_mapped = find_angle(coords_contrastive,fam[:,0],k = 20) # k = 20 used in paper
coords_popvae = pd.read_csv("saved_files/popvae_dog.txt", delimiter = "\t").to_numpy()[:,:2].astype(float)
coords_triplet = read_h5("saved_files/triplet_dog.h5", f"{5000}_encoded_train")
triplet_mapped = find_angle(coords_triplet,fam[:,0],k = 20) # k = 20 used in paper


# Human

coords_contrastive_human = read_h5("saved_files/centroid_human.h5", f"{5000}_encoded_train")
encoded_train_mapped_human = find_angle(coords_contrastive_human,fam_human[:,0],k = 20) # k = 20 used in paper


coords_contrastive_human_valmiss = read_h5("saved_files/centroid_human_masked.h5", f"{5000}_encoded_train")
encoded_train_mapped_human_valmiss = find_angle(coords_contrastive_human_valmiss,fam_human[:,0],k = 30) # k = 20 used in paper

coords_popvae_human = pd.read_csv("saved_files/popvae_human.txt", delimiter = "\t").to_numpy()[:,:2].astype(float)
coords_popvae_human_valmiss = pd.read_csv("saved_files/popvae_human_masked.txt", delimiter = "\t").to_numpy()[:,:2].astype(float)



#  PCA preprocessing results dog
umap_dog3 = np.load("saved_files/umap_tsne_pca/umap_dog3.npy")
umap_dog30 = np.load("saved_files/umap_tsne_pca/umap_dog30.npy")
tsne_dog3 = np.load("saved_files/umap_tsne_pca/tsne_dog3.npy")
tsne_dog30 = np.load("saved_files/umap_tsne_pca/tsne_dog30.npy")
pca_dog = np.load("saved_files/PCA_dog.npy")

# Compute all evaluation metrics for the given embeddings
coord_dict = {"Centroid":coords_contrastive,
              "Triplet": coords_triplet,
              "popvae":coords_popvae,
                "PCA":pca_dog,
                "t-SNE 3":tsne_dog3,
                "t-SNE 30":tsne_dog30,
              "UMAP 3 ": umap_dog3,
              "UMAP 30": umap_dog30,
              }

results_dict_dog = {}
for crd in coord_dict:
    print(crd)
    r_dict = get_eval_metrics(fam[:,0], spops, coord_dict[crd], reference_neighbors, train_inds, valid_inds, k_vec_ = [3], fixed_k_ = 30)
    results_dict_dog[crd] = r_dict

#with open('saved_files/result_dicts/results_dict_dog_PCA.pickle', 'wb') as handle:
#    pickle.dump(results_dict_dog, handle, protocol=pickle.HIGHEST_PROTOCOL)


# PCA HUMAN Masked

tsne3_human_valmiss = np.load(f"saved_files/umap_tsne_pca/tsne_human_valmiss{3}.npy")
tsne30_human_valmiss = np.load(f"saved_files/umap_tsne_pca/tsne_human_valmiss{30}.npy")
umapped_human_3_valmiss = np.load(f"saved_files/umap_tsne_pca/umap_human_valmiss{3}.npy")
umapped_human_30_valmiss = np.load(f"saved_files/umap_tsne_pca/umap_human_valmiss{30}.npy")
PCA_human_masked = np.load("saved_files/PCA_human_masked.npy")

coord_dict_human = {
            "Centroid":coords_contrastive_human_valmiss,
            "popvae":coords_popvae_human_valmiss,
            "PCA": PCA_human_masked,
            "t-SNE 3": tsne3_human_valmiss,
            "t-SNE 30": tsne30_human_valmiss,
            "UMAP 3": umapped_human_3_valmiss,
            "UMAP 30": umapped_human_30_valmiss,
}

results_dict_human_masked= {}
for crd in coord_dict_human:
    print(crd)
    r_dict = get_eval_metrics(fam_human[:,0], spops_human, coord_dict_human[crd], reference_neighbors_human, train_inds_human, valid_inds_human, k_vec_ = [3], fixed_k_ = 30)
    results_dict_human_masked[crd] = r_dict

#with open('saved_files/result_dicts/results_dict_human_masked_PCA.pickle', 'wb') as handle:
#    pickle.dump(results_dict_human_masked, handle, protocol=pickle.HIGHEST_PROTOCOL)



# PCA DATA FOR UMAP AND TSNE
tsne3_human = np.load(f"saved_files/umap_tsne_pca/tsne_human{3}.npy")
tsne30_human= np.load(f"saved_files/umap_tsne_pca/tsne_human{30}.npy")
umapped_human_3 = np.load(f"saved_files/umap_tsne_pca/umap_human{3}.npy")
umapped_human_30= np.load(f"saved_files/umap_tsne_pca/umap_human{30}.npy")
PCA_human = np.load("saved_files/PCA_human.npy")

coord_dict_human = {
            "Centroid":coords_contrastive_human,
            "popvae":coords_popvae_human,
            "PCA": PCA_human,
            "t-SNE 3": tsne3_human,
            "t-SNE 30": tsne30_human,
            "UMAP 3": umapped_human_3,
            "UMAP 30": umapped_human_30,
}

results_dict_human= {}
for crd in coord_dict_human:
    print(crd)
    r_dict = get_eval_metrics(fam_human[:,0], spops_human, coord_dict_human[crd], reference_neighbors_human, train_inds_human, valid_inds_human, k_vec_ = [3], fixed_k_ = 30)
    results_dict_human[crd] = r_dict

with open('saved_files/result_dicts/results_dict_human_PCA.pickle', 'wb') as handle:
    pickle.dump(results_dict_human, handle, protocol=pickle.HIGHEST_PROTOCOL)


results_dict_human_valmiss = pickle.load(open(f'saved_files/result_dicts/results_dict_human_masked_PCA.pickle', 'rb'))
results_dict_human = pickle.load(open(f'saved_files/result_dicts/results_dict_human_PCA.pickle', 'rb'))
results_dict_dog = pickle.load(open(f'saved_files/result_dicts/results_dict_dog_PCA.pickle', 'rb'))


# Print the table for the manuscript (TABLE 2).
# Dog
print("\\begin{table*}[p] \centering \label{table:popclass}")
print("\caption{Classification performance of the different methods using a KNN classifier with $k = 3$ on the embedding coordinates. Using a 80/20 train/test split for all methods except t-SNE.  }")
print("\\begin{tabularx}{\\textwidth}{@{}XXXXX@{}}")
print("\hline")
print( "{\\bf Method} & {\\bf Dataset} & {\\bf Superpop clustering(G)} & {\\bf Subpop acc.(L) }& {\\bf Validation (GE)} \\\\")
print("\hline")
for method_  in results_dict_dog:
    print(f" {method_} & Dog &  {results_dict_dog[method_]['glob_spop_umap_accuracy'][0]:.4f}   & {results_dict_dog[method_]['train_knn_acc']:.4f} &  {results_dict_dog[method_]['valid_knn_acc']:.4f}  \\\\")
print("\hline")

#Human
print( "{\\bf Method} & {\\bf Dataset} & {\\bf Superpop clustering(G)} & {\\bf Subpop acc.(L) }& {\\bf Validation (GE)} \\\\")
print("\hline")
for method_  in results_dict_human:
    print(f" {method_} & Human &  {results_dict_human[method_]['glob_spop_umap_accuracy'][0]:.4f}   & {results_dict_human[method_]['train_knn_acc']:.4f} &  {results_dict_human[method_]['valid_knn_acc']:.4f}  \\\\")
print("\hline")

#Human missing
print( "{\\bf Method} & {\\bf Dataset} & {\\bf Superpop clustering(G)} & {\\bf Subpop acc. (L) }& {\\bf Validation (GE)} \\\\")
print("\hline")

for method_  in results_dict_human_masked:
    print(f" {method_} & Human Masked &  {results_dict_human_masked[method_]['glob_spop_umap_accuracy'][0]:.4f}   & {results_dict_human_masked[method_]['train_knn_acc']:.4f} &  {results_dict_human_masked[method_]['valid_knn_acc']:.4f}  \\\\")
print("\end{tabularx}")
print(  "\label{tab:popclass}")
print("\end{table*}")



files = ['0_0.h5', '0_05.h5', '0_099.h5', '05_0.h5', '05_05.h5', '05_099.h5',
       '099_0.h5', '099_05.h5', '099_099.h5']
score = []
for i in files:
    coords_M1 = read_h5("saved_files/aug_abl/"+ i, f"{5000}_encoded_train")
    knn_score = knn_f1_tt(spops, coords_M1, train_inds, valid_inds, [3], verbose = False)[-1]
    score.append(knn_score)

fmt = ".6f"
print("\\begin{table} \n \\centering \n     \\begin{tabular}{c|ccc}")
print(f" p_flip \ p_mask & 0.0 & 0.5 & 0.99 \\\\")
print( "\hline ")
print(f"     0.0  & {score[0]:{fmt}} &{score[1]:{fmt}} & {score[2]:{fmt}} \\\\")
print(f"     0.5  & {score[3]:{fmt}} &{score[4]:{fmt}} & {score[5]:{fmt}} \\\\")
print(f"     0.99  & {score[6]:{fmt}} &{score[7]:{fmt}} & {score[8]:{fmt}} \\\\")
print(" \end{tabular} \n \caption{Caption} \n \label{tab:my_label} \n \end{table}")


# FIGURE  6
fig = plt.figure(figsize=(2*fsize, 3*fsize), linewidth=lw_figure)
plt.subplot(3,2,1)
plot_by_thing_kaus(fam, pca_dog, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("PCA",fontsize=10)
plt.subplot(3,2,2)
plot_by_thing_kaus(fam, tsne_dog30, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("t-SNE 30",fontsize=10)
plt.subplot(3,2,3)
plot_by_thing_kaus(fam, triplet_mapped, proj_fam_file_dog, spop_file_dog, plot_border= True, axisequal=False,createfig=False)
plt.title("Triplet",fontsize=10)
plt.subplot(3,2,4)
plot_by_thing_kaus(fam, encoded_train_mapped, proj_fam_file_dog, spop_file_dog, plot_border= True, axisequal=False,createfig=False)
plt.title("Centroid",fontsize=10)
plt.subplot(3,2,6)
plot_by_thing_kaus(fam, umap_dog3, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("UMAP 3 ",fontsize=10)
plt.subplot(3,2,5)
plot_by_thing_kaus(fam, coords_popvae, proj_fam_file_dog, spop_file_dog, plot_border= False, axisequal=False,createfig=False)
plt.title("popvae",fontsize=10)
plt.tight_layout()
plt.savefig("saved_files/plots/dog_embeddings.pdf")
plt.show()


# FIGURE 7 These take a while to create, consider saving them
K = 250
t1s = knn_f1(spops, tsne_dog30,np.arange(1,K))[1]
t2s = knn_f1(spops, tsne_dog3,np.arange(1,K))[1]
u1s= knn_f1(spops, umap_dog30,np.arange(1,K))[1]
u2s= knn_f1(spops, umap_dog3,np.arange(1,K))[1]
c1s= knn_f1(spops, coords_contrastive,np.arange(1,K))[1]
p1s= knn_f1(spops, pca_dog,np.arange(1,K))[1]
triplet1s= knn_f1(spops, coords_triplet,np.arange(1,K))[1]
popvae1s= knn_f1(spops, coords_popvae,np.arange(1,K))[1]

dict_to_save = {"t1s":t1s, "t2s":t2s, "u1s":u1s, "u2s":u2s, "c1s":c1s, "p1s":p1s, "triplet1s":triplet1s, "popvae1s":popvae1s}

with open('saved_files/result_dicts/fig7_dict.pickle', 'wb') as handle:
    pickle.dump(dict_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

loaded_dict = pickle.load(open(f'result_dicts/fig7_dict.pickle', 'rb'))


clist = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd','#9467bd',  '#8c564b','#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
stylelist = ["-","-","-","-","-","--","-","--"]
lw =0.5
plt.figure(figsize = (6,3))
plt.plot(c1s,color = clist[0],linestyle = stylelist[0],label = "Centroid", linewidth =lw )
plt.plot(triplet1s,color = clist[1],linestyle = stylelist[1],label = "Triplet", linewidth =lw )
plt.plot(popvae1s,color = clist[2],linestyle = stylelist[2],label = "popvae", linewidth =lw )
plt.plot(p1s,color = clist[3], linestyle = stylelist[3],label = "PCA", linewidth =lw )
plt.plot(t2s,color = clist[4], linestyle = stylelist[4],label = "t-SNE 3", linewidth =lw )
plt.plot(t1s,color = clist[5],linestyle = stylelist[5],label = "t-SNE 30", linewidth =lw )
plt.plot(u2s,color = clist[6], linestyle = stylelist[6],label = "UMAP 3", linewidth =lw )
plt.plot(u1s,color = clist[7],linestyle = stylelist[7],label = "UMAP 30", linewidth =lw )
plt.legend()
plt.xlabel("Number of neighbors K", fontsize = 10)
plt.ylabel("KNN classification score", fontsize = 10)
plt.savefig("saved_files/plots/large_K_knn.pdf")
plt.show()



# LOCAL preservation as function of neighbours considered Figure 8
plt.figure(figsize = (6,3))
k = 100
for count, method_  in enumerate(results_dict_dog):
    plt.plot(results_dict_dog[method_]["local_adherence_k_vec"][:k],results_dict_dog[method_]["local_adherence_score"][:k], color = clist[count], linestyle = stylelist[count],label  = method_, linewidth = lw)
plt.legend()
plt.xlabel('Number of neighbors k', fontsize = 10)
plt.ylabel("Ratio of overlap", fontsize = 10)
plt.savefig("saved_files/plots/local_preservation.pdf")
plt.show()


# FIGURE  9 in revised version
fig = plt.figure(figsize=(2*fsize, 3*fsize), linewidth=lw_figure)
plt.subplot(3,2,1)
plot_by_thing_kaus(fam_human, encoded_train_mapped_human_valmiss, proj_fam_file_human, spop_file_human, plot_border= False, axisequal=False,createfig=False,marker_size = 20)
plt.title("Centroid",fontsize=10)
plt.subplot(3,2,2)
scatter_92(encoded_train_mapped_human_valmiss, nine, train_inds_human, valid_inds_human, createfig=False, s = 30)
plt.subplot(3,2,3)
plot_by_thing_kaus(fam_human, tsne30_human_valmiss, proj_fam_file_human, spop_file_human, plot_border= False, axisequal=False,createfig=False,marker_size = 20)
plt.title("t-SNE 30",fontsize=10)
plt.subplot(3,2,4)
scatter_92(tsne30_human_valmiss, nine, train_inds_human, valid_inds_human, createfig=False, s = 30)
plt.subplot(3,2,5)
plot_by_thing_kaus(fam_human, coords_popvae_human_valmiss, proj_fam_file_human, spop_file_human, plot_border= False, axisequal=False,createfig=False,marker_size = 20)
plt.title("popvae",fontsize=10)
plt.subplot(3,2,6)
scatter_92(coords_popvae_human_valmiss, nine, train_inds_human, valid_inds_human, createfig=False, s = 30)
plt.tight_layout()
plt.savefig("saved_files/plots/Human_embeddings.pdf")
plt.show()






# SUPPLEMENTAL FIGURES AND TABLES:

# Spherical

umapped_3 = np.load("saved_files/umap_tsne_pca/umap_dog3.npy")
umapped_30 = np.load("saved_files/umap_tsne_pca/umap_dog30.npy")

umapped_sphere3 = np.load("saved_files/umap_tsne_pca/umap_sphere_3.npy")
umapped_sphere30 = np.load("saved_files/umap_tsne_pca/umap_sphere_30.npy")
encoded_train_mapped_umap3 = find_angle(umapped_sphere3,fam[:,0],k = 20) # k = 20 used in paper
encoded_train_mapped_umap30 = find_angle(umapped_sphere30,fam[:,0],k = 20) # k = 20 used in paper

g_ump2d_30 = global_spop(fam[:,0], spops,umapped_30 )[0]
g_ump2d_3 = global_spop(fam[:,0], spops,umapped_3 )[0]
g_ump3d_30 = global_spop(fam[:,0], spops,umapped_sphere30 )[0]
g_ump3d_3 = global_spop(fam[:,0], spops,umapped_sphere3 )[0]
g_cnt = global_spop(fam[:,0], spops,coords_contrastive )[0]

ump2d_3 = knn_f1_tt(fam[:,0], umapped_3,train_inds, valid_inds, [3], verbose = False)
ump2d_30 = knn_f1_tt(fam[:,0], umapped_30,train_inds, valid_inds, [3], verbose = False)
ump3d_30 = knn_f1_tt(fam[:,0], umapped_sphere30,train_inds, valid_inds, [3], verbose = False)
ump3d_3 = knn_f1_tt(fam[:,0], umapped_sphere3,train_inds, valid_inds, [3], verbose = False)
cnt3d = knn_f1_tt(fam[:,0], coords_contrastive,train_inds, valid_inds, [3], verbose = False)

print("%__ Table comparing 3D Spherical UMAP vs 2D default UMAP")
print("\\begin{tabularx}{\\textwidth}{@{}XXXXX@{}}")
print("\hline")
print( "{\\bf Method} & Neighbors & {\\bf Glob. score(G)} & {\\bf Subpop acc.(L) }& {\\bf Validation (GE)} \\\\")
print("\hline")
print(f"UMAP 2D & 3 & {g_ump2d_3[0]:.4f} &  {ump2d_3[0]:.4f} & {ump2d_3[1]:.4f}   \\\ ")
print(f"UMAP 2D & 30 &  {g_ump2d_30[0]:.4f} &  {ump2d_30[0]:.4f} & {ump2d_30[1]:.4f}  \\\ ")
print(f"UMAP Spherical &3 &  {g_ump3d_3[0]:.4f} &  {ump3d_3[0]:.4f} & {ump3d_3[1]:.4f}   \\\ ")
print(f"UMAP Spherical & 30 &  {g_ump3d_30[0]:.4f} &  {ump3d_30[0]:.4f} & {ump3d_30[1]:.4f}     \\\ ")
#print(f"Centroid & - &  {g_cnt[0]:.4f} &  {cnt3d[0]:.4f} & {cnt3d[1]:.4f}   \\\ ")
#print(f"Centroid &  - & \\textbf{0.5709 }&  \\textbf{0.4614} & \\textbf{0.9130} & \\textbf{0.9294 }\\\ ")

print("\hline\\")
print("\end{tabularx}")




# Centroid negative scaling image
anc = np.array([0,0])
pos = np.array([0,1])
neg = np.array([50,0])

centroid1 = np.mean(np.concatenate([anc[np.newaxis,:],pos[np.newaxis,:],neg[np.newaxis,:]], axis = 0), axis = 0)
centroid2 = np.sum(np.concatenate([anc[np.newaxis,:],pos[np.newaxis,:],2*neg[np.newaxis,:]], axis = 0), axis = 0) / 4


plt.figure()
plt.subplot(2,1,1)
plt.plot([pos[0], centroid1[0]], [pos[1], centroid1[1]], 'k')
plt.plot([neg[0], centroid1[0]], [neg[1], centroid1[1]], 'k' )
plt.plot([anc[0], centroid1[0]], [anc[1], centroid1[1]], 'k' )
plt.scatter(anc[0], anc[1], s = 50, c =  'b', label = "Anchor")
plt.scatter(pos[0], pos[1], s = 50, c =  'g', label = "Postive")
plt.scatter(neg[0], neg[1], s = 50, c =  'r', label = "Negative")
plt.scatter(centroid1[0], centroid1[1], s = 50, c =  'y' )
plt.legend()
plt.axis("equal")
plt.title("C = (A+P+N)/3", fontsize = 10)
plt.subplot(2,1,2)
plt.plot([pos[0], centroid2[0]], [pos[1], centroid2[1]], 'k')
plt.plot([neg[0], centroid2[0]], [neg[1], centroid2[1]], 'k' )
plt.plot([anc[0], centroid2[0]], [anc[1], centroid2[1]], 'k' )
plt.scatter(anc[0], anc[1], s = 50, c =  'b')
plt.scatter(pos[0], pos[1], s = 50, c =  'g')
plt.scatter(neg[0], neg[1], s = 50, c =  'r')
plt.scatter(centroid2[0], centroid2[1], s = 50, c =  'y' )
plt.axis("equal")
plt.title("C = (A+P+2N)/4", fontsize = 10)
plt.savefig("saved_files/plots/neg_scaling.pdf")
plt.show()



pca3 = PCA(n_components=3)
pca3.fit(geno_data_normed[:,train_inds].T)
X_PCA3 = pca3.transform(geno_data_normed.T)

# TEST t-SNE 3d PLOT
tsne_dog_3D = np.load("saved_files/umap_tsne_pca/tsne_emb3d.npy") # 30 perplexity

tsne3normed = tsne_dog_3D / np.sqrt(np.sum(tsne_dog_3D**2, axis = 1, keepdims=True))
pca3normed = X_PCA3 / np.sqrt(np.sum(X_PCA3**2, axis = 1, keepdims=True))
tsne3_mapped = find_angle(tsne3normed, fam[:,0])
pca3_mapped = find_angle(pca3normed, fam[:,0])


# FIGURE S3
plt.figure(figsize = (6,3))
plt.subplot(1,2,1)
plot_by_thing_kaus(fam,tsne3_mapped,proj_fam_file_dog, spop_file_dog,axisequal=False, plot_border = True, createfig = False)
plt.title("t-SNE 30",fontsize=10)
plt.tight_layout()
plt.subplot(1,2,2)

plot_by_thing_kaus(fam,pca3_mapped,proj_fam_file_dog, spop_file_dog,axisequal=False, plot_border = True, createfig = False)
plt.title("PCA",fontsize=10)
plt.tight_layout()
plt.savefig("saved_files/plots/pca_tsne_3D.pdf")



gtsne2d = global_spop(fam[:,0], spops,tsne_dog30 )[0]
gtsne3d = global_spop(fam[:,0], spops,tsne3_mapped )[0]
gpca2d = global_spop(fam[:,0], spops,pca_dog )[0]
gpca3d = global_spop(fam[:,0], spops,pca3_mapped )[0]


Ltsne2d = knn_f1_tt(fam[:,0], tsne_dog30,train_inds, valid_inds, [3], verbose = False)
Ltsne3d = knn_f1_tt(fam[:,0], tsne3_mapped,train_inds, valid_inds, [3], verbose = False)
Lpca2d = knn_f1_tt(fam[:,0], pca_dog,train_inds, valid_inds, [3], verbose = False)
Lpca3d = knn_f1_tt(fam[:,0], pca3_mapped,train_inds, valid_inds, [3], verbose = False)

print("__ Table comparing 3D t-SNE  and PCA + normalization and projection")
print("\\begin{tabularx}{\\textwidth}{@{}XXXXX@{}}")
print("\hline")
print( "{\\bf Method} & {\\bf Superpop clust.(G)} & {\\bf Subpop acc. (L) }& {\\bf Validation (GE)} \\\\")
print("\hline")
print(f"t-SNE 2D & {gtsne2d[0]:.4f} &  {Ltsne2d[0]:.4f} & {Ltsne2d[1]:.4f}   \\\ ")
print(f"t-SNE 3D + projection  &  {gtsne3d[0]:.4f} &  {Ltsne3d[0]:.4f} & {Ltsne3d[1]:.4f}  \\\ ")
print(f"PCA 2D  &  {gpca2d[0]:.4f} &  {Lpca2d[0]:.4f} & {Lpca2d[1]:.4f}   \\\ ")
print(f"PCA 3D + projection  &  {gpca3d[0]:.4f} &  {Lpca3d[0]:.4f} & {Lpca3d[1]:.4f}     \\\ ")
#print(f"Centroid &  - & \\textbf{0.5709 }&  \\textbf{0.4614} & \\textbf{0.9130} & \\textbf{0.9294 }\\\ ")

print("\hline\\")
print("\end{tabularx}")



#

dd_unfiltered = np.load("saved_files/distance_human_unfiltered.npy")
dd_filtered = np.load("saved_files/distance_human.npy")

centroid_L = local_adherence(reference_neighbors_human, coords_contrastive_human, kmax = 200, step = 1 )
popvae_L = local_adherence(reference_neighbors_human, coords_popvae_human, kmax = 200, step = 1 )
tsne_30_L = local_adherence(reference_neighbors_human, tsne30_human, kmax = 200, step = 1 )
tsne_3_L = local_adherence(reference_neighbors_human, tsne3_human, kmax = 200, step = 1 )
umap_3_L = local_adherence(reference_neighbors_human, umapped_human_3, kmax = 200, step = 1 )
umap_30_L = local_adherence(reference_neighbors_human, umapped_human_30, kmax = 200, step = 1 )

# Unfiltered dataset
nbrs2 = NearestNeighbors(n_neighbors=neibs, algorithm='ball_tree').fit(dd_unfiltered)
distances2, indices_embedding = nbrs2.kneighbors(dd_unfiltered)
score_embedding = []
k_vec = []


for k in np.arange(3, 200, 1):
    score_embedding_temp = 0

    for i in range(dd_unfiltered.shape[0]):
        score_embedding_temp += np.intersect1d(reference_neighbors_human[i, 1:k], indices_embedding[i, 1:k]).shape[0]

    score_embedding.append(score_embedding_temp / (reference_neighbors_human.shape[0] * len(reference_neighbors_human[i, 1:k])))

    k_vec.append(len(reference_neighbors_human[i, 1:k]))

import seaborn as sns
sns.set()
plt.figure(figsize = (12,6))
plt.plot(k_vec, score_embedding,label = "Unfiltered data")
plt.plot(k_vec,centroid_L[0], label  = "Centroid")
plt.plot(k_vec,popvae_L[0], label  = "Popvae")
plt.plot(k_vec,tsne_30_L[0], label  = "t-SNE 30")
plt.plot(k_vec,umap_30_L[0], label  = "UMAP 30")
plt.plot(k_vec,tsne_3_L[0], label  = "t-SNE 3")
plt.plot(k_vec,umap_3_L[0], label  = "UMAP 3")
plt.legend()
plt.xlabel("Number of neighbors")
plt.ylabel("Ratio of overlap in neighbor sets")
plt.savefig("saved_files/plots/unfiltered_local_score.pdf")
plt.show()


### TEST_NEW


dd_unfiltered = np.load("saved_files/distance_human_unfiltered.npy")
dd_filtered = np.load("saved_files/distance_human.npy")
embedding = coords_contrastive_human

def tester(reference_neighbors_human, embedding, kmax = 100, step = 2 ):

    d_embedding = distance_matrix(embedding,embedding)
    nbrs2 = NearestNeighbors(n_neighbors=2067, algorithm='ball_tree').fit(d_embedding)
    distances2, indices_embedding = nbrs2.kneighbors(d_embedding)

    score = []
    for i in range(1,kmax,step):
        #test = np.var(np.where(reference_neighbors_human[:,i,np.newaxis]  -  indices_embedding == 0)[1])
        test = np.sqrt(np.mean( (i -  np.where(reference_neighbors_human[:,i,np.newaxis]  -  indices_embedding == 0)[1] ) **2 ))
        score.append(test)
    return score

score_cont = tester(reference_neighbors_human, coords_contrastive_human)
score_popvae = tester(reference_neighbors_human, coords_popvae_human)
score_tsne30 = tester(reference_neighbors_human, tsne30_human)
score_tsne3 = tester(reference_neighbors_human, tsne3_human)
score_umap3 = tester(reference_neighbors_human, umapped_human_3)
score_umap30 = tester(reference_neighbors_human, umapped_human_30)
score_pca = tester(reference_neighbors_human, PCA_human)

# Unfiltered dataset
nbrs2 = NearestNeighbors(n_neighbors=2067, algorithm='ball_tree').fit(dd_unfiltered)
distances2, indices_embedding = nbrs2.kneighbors(dd_unfiltered)
score_embedding = []
k_vec = []
score = []
for i in range(1,100,2):
    k_vec.append(i)
    #test = np.var(np.where(reference_neighbors_human[:,i,np.newaxis]  -  indices_embedding == 0)[1])
    test = np.sqrt(np.mean((i - np.where(reference_neighbors_human[:, i, np.newaxis] - indices_embedding == 0)[1]) ** 2))

    score.append(test)



plt.figure()
plt.plot(k_vec,score, label = "Unfiltered")
plt.plot(k_vec,score_cont, label = "centroid")
plt.plot(k_vec,score_popvae, label = "popvae")
plt.plot(k_vec,score_tsne30, label = "tsne30")
plt.plot(k_vec,score_tsne3, label = "tsne3")
plt.plot(k_vec,score_umap3, label = "umap3")
plt.plot(k_vec,score_umap30, label = "umap30")
plt.plot(k_vec,score_pca, label = "pca")
plt.legend()
#plt.title("The variance of  ranking of the nth closest neighbor")
plt.title("The RMSE of ranking of the nth closest neighbor")
plt.xlabel("Number of neighbors n")
plt.ylabel("RMSE of embedding Rank ")

plt.savefig("saved_files/plots/RMSE_test.pdf")
plt.show()

#for i in range()
