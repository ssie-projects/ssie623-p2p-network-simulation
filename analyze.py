import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import numpy as np
# from pymotif import Motif
import random 

from enum import Enum

import utils.network_motif_counter as nmc

random.seed(315)
# G = nx.read_gpickle("./data/networks/N25_p_0_0_9.pickle")
# G_dht = nx.read_gpickle("./data/networks/N25_p_0_0_9_dht.pickle")

G = nx.read_gpickle("./data/networks/N100_p_0.09CONTENT_100.pickle")
G_dht = nx.read_gpickle("./data/networks/N100_p_0.09CONTENT_100_dht.pickle")

# label edges
for edge in G.edges:
    nx.set_edge_attributes(G, {edge: {"type": "PEER"}})

def plot_louvain_communities(_G, plot_title, write=True, filename="TEST", latex_format=False):
    """Plot Louvain communities"""
    # compute the best partition
    # partition = nx.algorithms.community.louvain_communities(_G)

    # cmap = cm.get_cmap('viridis', max([len(p) for p in partition]) + 1)
    # shapes = 'so^>v<dph8'

    # plt.figure(figsize=(8,8))
    # # draw the graph
    # pos = nx.spring_layout(_G)
    # # color the nodes according to their partition
    # # cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # for node, color in partition.items():
    #     nx.draw_networkx_nodes(G, pos, [node], node_size=100,
    #                         node_color=[cmap.colors[color]],
    #                         node_shape=shapes[color])
    plt.title("Connections between Louvain communities of DHT network")
    pos = nx.spring_layout(_G)
    nx.draw(_G, pos=pos, node_size=[100*_G.nodes[node]["size"] for node in _G], alpha=0.70)
    plt.savefig("./images/G_dht_N100_p_0_09_louvain.png")
    plt.show()

def plot_histogram(_series_lst, plot_title, write=True, filename="TEST", latex_format=False):
    """Generate a nice histogram"""
    # plt.figure(figsize=(14,7)) # Make it 14x7 inch
    fig = plt.figure(plot_title, figsize=(8, 8))
    plt.hist(_series_lst, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)

    plt.title(plot_title)
    plt.xlabel("Bins")
    plt.xlabel("Count")

    if write:
        img_dir = "./images/"
        plt.savefig(img_dir + filename + ".png", format="png")
    plt.show()
    
def plot_degree_rank_and_histogram(G, plot_title = "Degree of a random ER network", write=True, filename="image"):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)

    fig = plt.figure(plot_title, figsize=(8, 8))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax0 = fig.add_subplot(axgrid[0:3, :], aspect="equal")
    # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.circular_layout(G)
    # ax0 = fig.add_subplot(axgrid[3:, 2:])
    nx.draw_networkx_nodes(G, pos, ax=ax0, alpha=0.6, node_size=[d[1] * 1.25 for d in G.degree])
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.09)
    # ax0.bar(*np.unique(degree_sequence, return_counts=True))
    # ax0.set_title("Degree histogram of DHT")
    ax0.set_xlabel("Degree")
    ax0.set_ylabel("Number of Nodes")

    ax0.set_title(plot_title)
    ax0.set_axis_off()

    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")

    fig.tight_layout()
    if write:
        img_dir = "./images/"
        plt.savefig(img_dir + filename + ".png", format="png")
        # matplotlib.use("pgf")
        # matplotlib.rcParams.update({
        #     "pgf.texsystem": "pdflatex",
        #     'font.family': 'serif',
        #     'text.usetex': True,
        #     'pgf.rcfonts': False,   
        # })

        # plt.savefig(img_dir + filename + ".pgf", format="pgf")

    plt.show()

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def kbucket_dist(s1, s2, bits=256):
    """Calculate the kbucket distance between two bit strings"""
    for i in range(bits):
        dist = hamming2(s1, s2)
        if pow(2,i) <= dist < pow(2,i+1):
            return dist
        else:
            pass

if __name__ == "__main__":
    print("Peer network statistics")
    print(G)
    print("Median node degree: \t", np.median([G.degree[node] for node in G.nodes]))
    # average kbucket distance from content
    print("Avg. kbucket distance: \t", np.mean([G.edges[edge]["distance"] for edge in G.edges]))
    # centrality
    print("Avg. out degree centrality: \t", np.mean(list(nx.out_degree_centrality(G).values())))
    # closeness centrality
    print("Avg. closeness centrality: \t", np.mean(list(nx.closeness_centrality(G, distance="distance").values())))
    print("# unique content on network: \t", len(set([content for node in G.nodes for content in G.nodes[node]["pinned"]])))
    print("Median number of messages pinned: \t", np.median([len(G.nodes[node]["pinned"]) for node in G.nodes]))
    # plot_degree_rank_and_histogram(G, plot_title="Degree distribution of peer network", filename="G_N100_p_0_09", )
    # plot_degree_rank_and_histogram(
    #     G, 
    #     plot_title="Degree distribution of peer network", 
    #     filename="G_N100_p_0_09",
    #     write=False
    # )
    

    print("Distributed hash table statistics")
    print(G_dht)
    print("Median node degree: \t", np.median([G_dht.degree[node] for node in G_dht.nodes]))
    # average kbucket distance from content
    # print("Avg. kbucket distance: \t", np.mean([G_dht.edges[edge]["distance"] for edge in G_dht.edges]))
    # centrality
    print("Avg. out degree centrality: \t", np.mean(list(nx.out_degree_centrality(G_dht).values())))
    # closeness centrality
    print("Avg. closeness centrality: \t", np.mean(list(nx.closeness_centrality(G_dht, distance="distance").values())))
    
    # plot_degree_rank_and_histogram(
    #     G_dht, 
    #     plot_title="Degree distribution of DHT network", 
    #     filename="G_dht_N100_p_0_09",
    #     write=False
    # )
    # total number of available content in the network
    content_count = len(set([content for node in G.nodes for content in G.nodes[node]["pinned"]]))
    print("Median DHT lengths: \t", np.median([len(G.nodes[node]["pinned"]) for node in G.nodes]))
    print("Average proportion of CID awareness: \t", np.mean([len(G.nodes[node]["pinned"]) / content_count for node in G.nodes]))
    dht_lengths = [len(G.nodes[node]["pinned"]) for node in G.nodes]
    plot_histogram(
        dht_lengths,
        "Number of CIDs tracked on Distributed Hash Table by each node",
        filename="G_dht_N100_p_0_09_CONTENT_100",
        write=True
    )

    # compute the best partition
    print("Louvain community analysis: \t")

    # Create Louvain community graph of DHT
    lc_G_dht = nx.algorithms.community.louvain_communities(G_dht)
    G_dht_lc = nx.DiGraph()

    for i in range(len(lc_G_dht)-2):
        for j in range(1,len(lc_G_dht)-1):
            louvain_group_pairs = list(set([(a,b) for a in lc_G_dht[i] for b in lc_G_dht[j]]))
            # print("Louvain group: ", lc_G_dht[i])
            # print("Louvain group: ", lc_G_dht[j])
            # print("Edge weights between Louvain group ", i, " and ", j, " : \t", len(louvain_group_pairs))
            edge_weight = len(louvain_group_pairs)
            G_dht_lc.add_nodes_from([i], size = len(lc_G_dht[i]))
            G_dht_lc.add_nodes_from([j], size = len(lc_G_dht[j]))            
            G_dht_lc.add_edges_from([(i,j)], weight=edge_weight)
            # now do the reverse since this is a directed graph
            # del louvain_group_pairs
            # louvain_group_pairs = list(set([(b,a) for a in lc_G_dht[i] for b in lc_G_dht[j]]))
            # print("Louvain group: ", lc_G_dht[i])
            # print("Louvain group: ", lc_G_dht[j])
            # print("Edge weights between Louvain group ", i, " and ", j, " : \t", len(louvain_group_pairs))
            # edge_weight = len(louvain_group_pairs)            
            # G_dht_lc.add_edges_from([(j,i)], weight=edge_weight)
            
            # print(louvain_group_pairs)
    print("Avg. closeness centrality of Louvain community: \t", np.mean(list(nx.closeness_centrality(G_dht_lc, distance="distance").values())))
    plot_louvain_communities(G_dht_lc, "Louvain communities")

    print("Motif analysis: G", G)
    # G_motifs = nmc.mcounter(G, nmc.motifs)
    # print(G_motifs)

    # fig, axs = plt.subplots(2,3, figsize=(15,15))

    # for i, g in enumerate(list(nmc.motifs.values())):
    #     nx.draw_networkx(g, ax=axs[i])
    # plt.show()
    # ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    # ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    # ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    # ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
    # ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)

    # ax1.plot(nmc.motifs["S1"])
    # ax2.plot(nmc.motifs["S2"])

    # fig, axes = plt.subplots(nrows=2, ncols=3)
    # ax = axes.flatten()

    # for i in range(len(nmc.motifs)):
    #     nx.draw_networkx(nmc.motifs[i], ax=ax[i])
    #     ax[i].set_axis_off()

    # plt.show()
    
    # motif = Motif()
    # motif.plot()
    # plot_degree_rank_and_histogram(G_dht, "Degree distribution of DHT network")
    # Louvain communities
    print(nx.algorithms.community.louvain_communities(G))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos=pos)
    
    
    # plt.show()