# this script generates a set of arbitrary bitstring records
# to represent content being requested and shared over a p2p network
import random
random.seed(315)

nbits = 256
nnodes = 100

if __name__ == "__main__":
    with open("./data/nodeids.txt", "w") as f:
        for i in range(nnodes):
            d = f'{random.getrandbits(nbits):=0{int(nbits)}b}'
            print(len(d))
            f.write('{}\n'.format(d))
    f.close()