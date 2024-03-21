import os.path

import faiss
import getopt
import numpy as np
import sys
import logging

logging.getLogger(__name__).setLevel(logging.INFO)


def loadGraphFromFile(graphFile: str):
    if os.path.isfile(graphFile) is False:
        logging.error(f"The path provided: {graphFile} is not a file")
        sys.exit(0)

    return faiss.read_index(graphFile)


def runSearch(index: faiss.Index):
    xq = np.array([[90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
                    132, 133, 134, 135, 136, 137, 138, 139]]).astype('float32')
    k = 10
    D, I = index.search(xq, k)

    # D and I are 2D arrays
    print(D[0][:k])  # returns top K element distance
    print(I[0][:k])  # returns top k element Id


def main(argv):
    opts, args = getopt.getopt(argv, "", ["graph_input_file="])
    graphInputFile = "/Volumes/workplace/VectorSearchForge/cagraindex-test-with-ids.txt"
    for opt, arg in opts:
        if opt == '-h':
            print('--graph_input_file <inputfile>')
            sys.exit()
        elif opt in "--graph_input_file":
            graphInputFile = arg

    index = loadGraphFromFile(graphInputFile)
    runSearch(index)


if __name__ == "__main__":
    main(sys.argv[1:])
