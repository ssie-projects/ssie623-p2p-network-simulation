from pymerkle import MerkleTree
import networkx as nx
from collections import defaultdict
import string
import random
from enum import Enum
import numpy as np

import matplotlib.pyplot as plt

# set random seed
random.seed(315) # setting seed for testing

# global variables
global G, G_dht
content_count = 100
content_lst = [f'{random.getrandbits(256):=0256b}' for c in range(content_count)]

# parameters
alpha = 3
k = 20 # max size of each k-bucket
p_active_inactive = 0.05 # Baseline probability of a node going inactive
p_inactive_active = 0.55
p_unpin_content = 0.05
p_pin_cached_content = 0.25

# node states
class State(Enum):
    INACTIVE = 0
    ACTIVE = 1

# Number of server nodes in the network
N = 1000
p = 0.09
# Directed graph for message routing
# Edge from node n -> m means that m is a peer and n will ping m for messages.
G = nx.erdos_renyi_graph(
    n=N,
    p=p,
    directed=True
)

# Empty directed graph to track information routing
G_info = nx.DiGraph()
G_dht = nx.DiGraph()

## Functions.
# rand bit string chunking function
def chunkKbits(binary_string, b):
    """
    Accepts a binary string (eg '101000101') and returns a list of b sized chunks of that string.
    """
    chunks = []
    start = 0 # initalize the starting position to begin chunking from
    bitCounter = 0 # initialize counter used to slice bits
    for i, bit in enumerate(binary_string):
        bitCounter += int(bit)
        # if chunk size is met or there are less than `b` bit remaining in the string to chunk
        if sum([int(bit) for bit in binary_string[start:]]) < b:
            chunks.append(binary_string[start:])
            break
        elif bitCounter == b:
            # print(binary_string[start:i])
            chunks.append(binary_string[start:i+1])
            start = i+1
            bitCounter = 0
        # print(i, bit, "bitCounter:\t", bitCounter)
    # confirm that the chunks aggregate to the initial bitstring
    assert (''.join(chunks) == binary_string), "chunks represent original bitstring"
    return chunks

def hamming2(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def xor(id1, id2):
    """ Calculates the xor distance between two hash strings"""
    id1b = bin(int(id1))
    id2b = bin(int(id2))
    print("len id1b:\t", len(id1b), "\tid1b: \t", id1b)
    print("len id2b: \t", len(id2b), "\tid2b: \t", id2b)
    
    dist = hamming2(id1b, id2b)
    return dist

def node_lookup(G, requestor_node, alpha):
    """
    Implements the Kademlia node lookup algorithm .
    """
    pass
    # requestor_node references its own dht (k-buckets) to find

def unpin(node, prob):
    """
    With some probability the node can decide to unpin content.
    """
    unpin_idx_lst = []
    for cidx in range(len(node["pinned"])-1):
        if random.random() < prob:
            unpin_idx_lst.append(cidx)
        else:
            pass
    node["pinned"] = list(np.delete(node["pinned"], tuple(unpin_idx_lst)))
def churn(_G, node):
    """
    Removes a node from the network entirely.
    """
    _G.remove_node(node)

def become_inactive(node, prob):
    """
    With some probability an active node becomes inactive (goes offline).
    """
    if random.random() < prob:
        node["state"] == State.INACTIVE

def become_active(node, prob):
    """
    With some probability an inactive node becomes active (goes online).
    """
    if random.random() < prob:
        node["state"] == State.ACTIVE

# Kademlia RPCs
def ping(node):
    """
    Pings a node to learn if its state is ACTIVE or INACTIVE.
    """
    return node["state"]

def store():
    pass

def find_nearest_node(G, _node, _content, count=0):
    """
    A recursive function that traverses a graphs neighbors until the node
    with smallest nodeId <> content distance is found.
    """
    # if the algorithm has made 10 hops
    if count == 10:
        return _node
    
    # if the node has no neighbors
    if len(list(G.neighbors(_node))) < 1:
        return _node

    content_dist = hamming2(G.nodes[_node]["nodeId"], _content[1])

    peer_content_dist_lst = [hamming2(G.nodes[peer]["nodeId"], _content[1]) for peer in G.neighbors(_node)]
    min_peer_content_dist = min(peer_content_dist_lst)

    # if the current node is closest to the content, return current node.
    if content_dist < min_peer_content_dist:
        print("nearest node found: ", _node)
        return _node
    else:
        # send this message to nearest peer node.
        next_peer_node = list(G.neighbors(_node))[peer_content_dist_lst.index(min_peer_content_dist)]
        return find_nearest_node(G, next_peer_node, _content, count+1)

def find_value(G, node, value, _k=20, _alpha=3, count=0):
    """
    Traverses the network according to the Kademlia algorithm.

    G: the NetworkX graph to traverse
    node: node seeking the value
    value: bitstring entry being searched for
    _k: number of peers to seek (default=20)
    _alpha: number of requests sent to peers at once (default=3)
    """
    # check DHT first for the value
    try:
        # find first matching element in node DHT
        matching_idx = [content[1] for content in G.nodes[node._data]["dht"]].index(v)
        matching_node_id = G.nodes[node._data]["dht"][matching_idx][0]
        return find_value(G, G.nodes(matching_node_id), value, _k, _alpha, count+1)
    except:
        pass

    if count == 100:
        print("too many hops couldn't find the file")
        return False, False, count
    kbucket_peers = [bucket for bucket in G.nodes[node._data]["kbuckets"].values() if len(bucket) > 0]
    for kbucket in kbucket_peers:
        for peer in kbucket:
            if ping(G.nodes[peer]) == State.ACTIVE:
                # peer updates its own DHT with node details
                dist = hamming2(G.nodes[node._data]["nodeId"], G.nodes[peer]["nodeId"])
                for i in range(256):
                    if pow(2,i) <= dist < pow(2,i+1):
                        # if the node is already in the peer's k-bucket the node is moved to the first index
                        try:
                            # node is currently in the peer's k-bucket
                            nIdx = G.nodes[peer]["kbuckets"][i].index(node._data)
                            # move this node, the requestor, to the front of it's k-bucket list
                            G.nodes[peer]["kbuckets"][i].insert(0, G.nodes[peer]["kbuckets"][i].pop(nIdx))
                        except ValueError:
                            # # node is not currently in the peer's' k-bucket
                            # # if there are fewer than k entries, insert into the beginning
                            if len(G.nodes[peer]["kbuckets"][i]) < _k:
                                G.nodes[peer]["kbuckets"][i].insert(0, node._data)
                                # Create a graph edge
                                G.add_edges_from([(peer, node._data)])
                            else: # full k-bucket
                                # ping the least recently seen node, if active, move that node to the front
                                if ping(G.nodes[G.nodes[peer]["kbuckets"][i][-1]]) == State.ACTIVE:
                                    # if least recently seen is active it's moved to most recently seen (index 0) and the requestor node info is dropped
                                    G.nodes[peer]["kbuckets"][i].insert(0, G.nodes[peer]["kbuckets"][i].pop(len(G.nodes[peer]["kbuckets"][i])-1))
                                else:
                                    # remove that node from the dht
                                    del G.nodes[peer]["kbuckets"][i][-1]
                                    # remove the edges of that node from the graph
                                    G.remove_edges_from([(peer, G.nodes[peer]["kbuckets"][i][-1])])
                                    # insert the requestor node
                                    G.nodes[peer]["kbuckets"][i].insert(0, node._data)
                            pass
                        break
                    else:
                        pass
                # if peer has the record it returns it to the node's cache
                # otherwise it continues routing the request.
                try: # see if the peer node has the file
                    cidx = (G.nodes[peer]["pinned"] + G.nodes[peer]["cached"]).index(value)
                    # print("Value ", value, " found on node", peer)
                    return peer, (G.nodes[peer]["pinned"] + G.nodes[peer]["cached"])[cidx], count
                except ValueError:
                    # print("value not found with node ", peer, ".. will keep searching")
                    # recursion
                    find_value(G, G.nodes(peer), value, _k, _alpha, count+1)
            else: # peer is inactive
                # node updates its kbuckets
                # identify which k bucket index the now inactive peer is in
                kidx = [v.count(peer) for v in G.nodes[node._data]["kbuckets"].values()].index(1)
                # remove that inactive node from the node's k bucket
                G.nodes[node._data]["kbuckets"][kidx].remove(peer)

                # remove edges from the graph
                G.remove_edges_from([(node._data, peer)])
    
    # kbuckets exhausted, value not found
    return False
# https://stackoverflow.com/a/72419563

def initialize():
    """
    graph_setting: if "read" then 
    """
    global G, G_dht, content_lst

    for node in G.nodes():
        # create nodeId 256 bit string (will be used as PeerId)
        G.nodes()[node]["nodeId"] = f'{random.getrandbits(256):=0256b}'
        G.nodes()[node]["state"] = State.ACTIVE # nodes can randomly become inactive (go offline), in which case this becomes False
        G.nodes()[node]["activeSteps"] = 0 # iterate this each simulation step
        G.nodes()[node]["pinned"] = random.sample(content_lst, random.randint(0, 10))
        G.nodes()[node]["cached"] = []
        G.nodes()[node]["dht"] = [] # lists (NodeID, contentID) for all known key pairs
        
        # probability of going offline should be inversely proportional to number of consecutive active steps
    del node # clear node variable just in case

    # ensure all nodeIds are unique. if not, replace duplicates and check again.
    # counts the number of unique nodeIds and compares them to the number of nodes
    # these values should match. otherwise there is a duplicate to fix
    nodeId_lst = [G.nodes[node]["nodeId"] for node in G.nodes]

    while len(nodeId_lst) != N:
        dupes = defaultdict(list)
        for i,item in enumerate(nodeId_lst):
            dupes[item].append(i)
        dupes = {k:v for k,v in dupes.items() if len(v)>1}

        # cycle through duplicates and replace them with new random generated bit strings
        for item in list(dupes.items()):
            for n in range(len(item[1])-1): # this is the list of node indexes that have the duplicate nodeIds
                G.nodes[item[1][n]]["nodeId"] = f'{random.getrandbits(256):=0256b}'
        # update the list of NodeIds currently in the network
        nodeId_lst = [G.nodes[node]["nodeId"] for node in G.nodes]
        print("round of deduping complete")
        print("current # of unique ids: /t", len(nodeId_lst), " and current # nodes: \t", N)

    print("\u2713 ", N, " nodes with ", len(set([G.nodes[node]["nodeId"] for node in G.nodes])), " unique NodeIds")

    #################
    # distributd hash table
    #################
    # each node publishes their availability content to its peers' DHT
    for node in G.nodes:
        # pinned and cached content
        available_content_lst = [(node, content) for content in G.nodes[node]["pinned"] + G.nodes[node]["cached"]]
        for peer in G.neighbors(node):
            # this is a simplification, in IPFS this hosting info is routed until the closest NodeId is discovered, and it's stored with that node.
            
            # G.nodes[peer]["dht"].extend(available_content_lst)

            for content in available_content_lst:
                # route (key, value) message to closest available peer in the network
                # if one of the peer's peers has a closer nodeId send this message to them
                # otherwise update the peer dht
                # peer_content_dist = hamming2(G.nodes[peer]["nodeId"], content[1])
                # peer_peers_content_dist_lst = [hamming2(G.nodes[peer]["nodeId"], content[1]) for peer in G.neighbors(node)]
                # min_peer_peers_content_dist = min(peer_peers_content_dist_lst)

                # # if this peer has a nodeId with least distance to the content bitstring then update the peer's dht
                # if peer_content_dist < min_peer_peers_content_dist:
                #     G.nodes[peer]["dht"].extend(available_content_lst)
                #     break
                # # otherwise the peer relays the message to its peers until the node with least distance between its nodeId and the content is found.
                # else:
                #     min_peer_peers_content_dist_id = list(G.neighbors(peer))[peer_peers_content_dist_lst.index(min_peer_peers_content_dist)]
                print("Finding nearest node to content: \t", content)

                nearest_node = find_nearest_node(G, peer, content, 0)
                
                print("Nearest node identified: \t", nearest_node)
                print("Nearest node nodeId: \t", G.nodes[nearest_node]["nodeId"])
                
                # Constructing DHT network
                G_dht.add_nodes_from([(node, G.nodes[node])])
                G_dht.add_nodes_from([(nearest_node, G.nodes[nearest_node])])
                
                G_dht.add_edge(nearest_node, node, type="DHT")
                
                if tuple(content) not in G.nodes[nearest_node]["dht"]:
                    G.nodes[nearest_node]["dht"].append(tuple(content))
                else:
                    pass

    
    #################
    # routing table 
    #################
    # k buckets aka peerset
    # initial peerset is this list of neighbors from the randomly generated small-world network graph

    for node in G.nodes:
        # initialize an empty k-bucket dht from peerset (neighbors)
        G.nodes[node]["kbuckets"] = dict()
        for i in range(256):
            G.nodes[node]["kbuckets"][i] = list()

        # assign neighbors to k-bucket in the dht depending on distance between `NodeId` bitstrings (per Kademlia algorithm)
        # this list will eventually be sorted, with least recently seen nodes in the beginning
        # and most recently seen nodes at the end
        for neighbor in G.neighbors(node):
            # xor distance (simplified as hamming distance for bitstrings for this research)
            dist = hamming2(G.nodes[node]["nodeId"], G.nodes[neighbor]["nodeId"])
            for i in range(256):
                if pow(2,i) <= dist < pow(2,i+1):
                    G.nodes[node]["kbuckets"][i].append(neighbor)
                    nx.set_edge_attributes(G, {(node, neighbor): {"distance": i}})
                    break
                else:
                    pass

# bootstrapping
# preferential attachment to long lived nodes 
def update():
    global G, G_info
    # Each node broadcasts to their peers which Content Ids it is hosting.
    # Those broadcasts are routed through the network until the NodeId with the least distance to the ContentId
    # is found, then the message is added to that node's DHT.

    # Send message to closest peer.
        # If closest peer has a closer peer it routes the request to update the DHT to that node.
        # Else
        # Update DHT and k-buckets

    # Node changes for each step.
    # Nodes go inactive and active (simulating going offline and back online)
    for node in G.nodes:
        # if active, potentially go offline
        if G.nodes[node]['state'] == State.ACTIVE:
            if random.random() < p_active_inactive:
                G.nodes[node]['state'] = State.INACTIVE
        # if inactive, potentially come back online
        elif G.nodes[node]['state'] == State.INACTIVE:
            if random.random() < p_inactive_active:
                G.nodes[node]['state'] = State.ACTIVE

        # Some nodes churn and leave the network entirely
        # to be implemented later

def observe():
    global G, G_info
    print("Num active nodes: \t", len([1 for n in G.nodes if G.nodes[n]["state"] == State.ACTIVE]))
    # Nodes choose to pin some of their cached data (make permanent)

    # Active nodes broadcast their pinned and cached content to their peers

        # nodes have pinned and cached data
        ## cached data is the file they requested
        ## cached data is deleted (garbage collected) after some number of steps

if __name__ == "__main__":
    initialize()
    print(G)

    # write to file
    # change states
    # G
    for node in G.nodes:
        G.nodes[node]["state"] = G.nodes[node]["state"]._name_
    # G DHT
    for node in G_dht.nodes:
        G_dht.nodes[node]["state"] = G_dht.nodes[node]["state"]._name_

    print("writing G to file.")
    data_dir = "./data/networks/"
    G_filename = data_dir + "N" + str(N) + "_p_" + str(p) + "CONTENT_" + str(content_count)
    nx.write_gpickle(G, G_filename + ".pickle")

    print("writing G_dht to file.")
    G_dht_filename = data_dir + "N" + str(N) + "_p_" + str(p) + "CONTENT_" + str(content_count) + "_dht"
    nx.write_gpickle(G_dht, G_dht_filename + ".pickle")

    # Generate G and DHT graphs for 10, 100, 1000 nodes for
    # p = 0.05, p = .25, p =.75
    # 
    # Compare connectivity between the two
    #  
    # unpin(G.nodes[0], p_unpin_content)
    # for i in range(10):
    #     # observe()
    #     # update()

    #     # request = random.choice(content_lst)
    #     request = "0010101001111000011110010001101100101110111100110001110011100011110011000010111101011000111111111111010000100101111000111111111001011001100100000000101100010111010011000101010010000110010000100111001111100111000011010111011100001000000010011001110110010111"
    #     # requestor = G.nodes(random.randint(0, N))
    #     requestor = G.nodes(2)
    #     print("Node ", requestor, " requesting content ", request, "from the network")
    #     peer, value, hops = find_value(
    #         G, 
    #         requestor, 
    #         request,
    #         k,
    #         alpha,
    #         count=0
    #     )
    # request = "0001101000111101111011100010000000110010001001011100000001011110010101100000110010110000110000000111111110101011001110100000000101000101001111000100101110101010111010100000000000101011110001000100001011111011010010101001111111011000110110011011111010110111"
    # request randomly from the network
    # for i in range(5):
    #     request = random.sample([content for node in G.nodes for content in G.nodes[node]["pinned"] + G.nodes[node]["cached"]],1)[0]
    #     # confirm this request is available on the network
    #     print(request in [content for node in G.nodes for content in G.nodes[node]["pinned"] + G.nodes[node]["cached"]])
    #     # requestor = G.nodes(random.randint(0, N))
    #     requestor = G.nodes(random.sample(range(N),1)[0])
    #     print("Node ", requestor, " requesting content ", request, "from the network")
    # nx.plot(G)
    
    # peer, value, hops = find_value(
    #     G, 
    #     requestor, 
    #     request,
    #     k,
    #     alpha,
    #     count=0
    # )
    # if peer:
    #     G_info.add_edges_from([(1, peer)])
    # print(G_info)
    # print(peer, value, hops)
    # print(len(G.edges))
    
### Scrap
# how to get binary representation of nodeId
# bin(int(G.nodes()[node]["nodeId"]))

# DHT hashing and routing https://stackoverflow.com/a/59671257