from pymerkle import MerkleTree
import networkx as nx
from collections import defaultdict
import string
import random
from enum import Enum
import numpy as np

# parameters
alpha = 3

class State(Enum):
    INACTIVE = 0
    ACTIVE = 1
# tree = MerkleTree()
# N = 100
# data = [] # store the set of random content

# rand bit string function

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

# Number of server nodes in the network
N = 100

# Baseline probability of a node going inactive
p_active_inactive = 0.15
p_inactive_active = 0.02
p_unpin_content = 0.05
p_pin_cached_content = 0.25

# Directed graph
# Edge from node n -> m means that m is a peer and n will ping m for messages.
G = nx.erdos_renyi_graph(
    n=N,
    p=0.06,
    directed=True
)

# set of globally available content
# eventually will refactor this.
random.seed(315) # setting seed for testing
content_lst = [f'{random.getrandbits(256):=0256b}' for c in range(10)]

for node in G.nodes():
    # create nodeId 256 bit string (will be used as PeerId)
    G.nodes()[node]["nodeId"] = f'{random.getrandbits(256):=0256b}'
    G.nodes()[node]["state"] = State.ACTIVE # nodes can randomly become inactive (go offline), in which case this becomes False
    G.nodes()[node]["activeSteps"] = 0 # iterate this each simulation step
    G.nodes()[node]["pinned"] = random.sample(content_lst, random.randint(0, 10))
    G.nodes()[node]["cached"] = []
    
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
# routing table 
#################
# k buckets aka peerset
# initial peerset is this list of neighbors from the randomly generated small-world network graph

## all the nodes (by nodeId) that a request is sent to
k = 20

for node in G.nodes:
    # initialize an empty k-bucket dht from peerset (neighbors)
    G.nodes[node]["dht"] = dict()
    for i in range(256):
        G.nodes[node]["dht"][i] = list()

    # assign neighbors to k-bucket in the dht depending on distance between `NodeId` bitstrings (per Kademlia algorithm)
    # this list will eventually be sorted, with least recently seen nodes in the beginning
    # and most recently seen nodes at the end
    for neighbor in G.neighbors(node):
        # xor distance (simplified as hamming distance for bitstrings for this research)
        dist = hamming2(G.nodes[node]["nodeId"], G.nodes[neighbor]["nodeId"])
        for i in range(256):
            if pow(2,i) <= dist < pow(2,i+1):
                G.nodes[node]["dht"][i].append(neighbor)
                break
            else:
                pass
# xor distance

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

def find_node():
    pass

def find_value(G, node, value, _k=20, _alpha=3):
    """
    Traverses the network according to the Kademlia algorithm.

    G: the NetworkX graph to traverse
    node: node seeking the value
    value: bitstring entry being searched for
    _k: number of peers to seek (default=20)
    _alpha: number of requests sent to peers at once (default=3)
    """
    kbucket_peers = [bucket for bucket in G.nodes[node._data]["dht"].values() if len(bucket) > 0]
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
                            nIdx = G.nodes[peer]["dht"][i].index(node._data)
                            # remove this node, the requestor, to the front of it's k-bucket list
                            G.nodes[peer]["dht"][i].insert(0, G.nodes[peer]["dht"][i].pop(nIdx))
                        except ValueError:
                            # # node is not currently in the peer's' k-bucket
                            # # if there are fewer than k entries, insert into the beginning
                            if len(G.nodes[peer]["dht"][i]) < _k:
                                G.nodes[peer]["dht"][i].insert(0, node._data)
                            else: # full k-bucket
                                # ping the least recently seen node, if active, move that node to the front
                                if ping(G.nodes[G.nodes[peer]["dht"][i][-1]]) == State.ACTIVE:
                                    # if least recently seen is active it's moved to most recently seen (index 0) and the requestor node info is dropped
                                    G.nodes[peer]["dht"][i].insert(0, G.nodes[peer]["dht"][i].pop(len(G.nodes[peer]["dht"][i])-1))
                                else:
                                    # remove that node
                                    del G.nodes[peer]["dht"][i][-1]
                                    # insert the requestor node
                                    G.nodes[peer]["dht"][i].insert(0, node._data)
                            pass
                        break
                    else:
                        pass
                # if peer has the record it returns it to the node's cache
                # otherwise it continues routing the request.
                try: # see if the peer node has the file
                    cidx = (G.nodes[peer]["pinned"] + G.nodes[peer]["cached"]).index(value)
                    print("Value found!!!")
                    return (G.nodes[peer]["pinned"] + G.nodes[peer]["cached"])[cidx]
                except ValueError:
                    # recursion
                    find_value(G, G.nodes(peer), value, _k, _alpha)
            else: # peer is inactive
                # node updates its dht
                G.nodes[node._data]["dht"][i].remove(peer)
    # lookup the node dht
    # G.nodes[node._data]["dht"]
    G.nodes[node._data]["dht"]
# https://stackoverflow.com/a/72419563

# bootstrapping
# preferential attachment to long lived nodes 
if __name__ == "__main__":
    # for i in range(100):
    #     chunkKbits(
    #         binary_string=f'{random.getrandbits(256):=0256b}',
    #         b=16
    #     )
    # xor_data = []
    # xor_dist = xor(G.nodes[0]["nodeId"], G.nodes[1]["nodeId"])
    # xor_data.append(xor_dist)
    # print(xor_data)
    # simulation time steps should be 1 hour to coincide with parameter k in kademlia paper
    # for node in range(len(G.nodes())-1):
        # print(G.nodes()[node])
        # print(int(G.nodes()[node]["nodeId"]) ^ int(G.nodes()[node+1]["nodeId"]))
    unpin(G.nodes[0], p_unpin_content)

    find_value(
        G, 
        G.nodes(1), 
        "0001100001011111110110000011110000001010101111110100100011000111001110100110000100000000101111001011101111100011101001100001011001010001110100101100101100000111101001010000111110101001101011010000011111111011111011010110010101110100101100111011111101011101",
        k,
        alpha
    )
    
    for node in G.nodes:
        # if active, potentially go offline
        if G.nodes[node]['state'] == State.ACTIVE:
            if random.random() < p_active_inactive:
                G.nodes[node]['state'] == State.INACTIVE
        # if inactive, potentially come back online
        elif G.nodes[node]['state'] == State.INACTIVE:
            if random.random() < p_inactive_active:
                G.nodes[node]['state'] == State.ACTIVE
        
        # nodes broadcast all of the content CIDs they currently have available to their peers

        # nodes have pinned and cached data
        ## cached data is the file they requested
        ## cached data is deleted (garbage collected) after some number of steps




    # for i in range(10):
    #     # generate 256 bit integer for Node IDs as in the IPFS protocol
    #     data = str(random.getrandbits(256))


### Scrap
# how to get binary representation of nodeId
# bin(int(G.nodes()[node]["nodeId"]))

# DHT hashing and routing https://stackoverflow.com/a/59671257