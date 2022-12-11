from pymerkle import MerkleTree
import networkx as nx
from collections import defaultdict
import string
import random
import enum
import numpy as np

class State:
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
    G.nodes()[node]["active"] = State.ACTIVE # nodes can randomly become inactive (go offline), in which case this becomes False
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

def find_value():
    pass

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