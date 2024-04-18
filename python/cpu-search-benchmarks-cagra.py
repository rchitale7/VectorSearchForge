import os.path

import faiss
import getopt
import numpy as np
import sys
import logging
from python.dataset.dataset import HDF5DataSet, Context
from python.decorators.timer import timer_func
from python.utils.common_utils import recall_at_r

logging.getLogger(__name__).setLevel(logging.INFO)


def loadGraphFromFile(graphFile: str):
    if os.path.isfile(graphFile) is False:
        logging.error(f"The path provided: {graphFile} is not a file")
        sys.exit(0)

    return faiss.read_index(graphFile)


@timer_func
def prepare_data(datasetFile: str):
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.QUERY)
    xq: np.ndarray = index_dataset.read(index_dataset.size())
    gt = HDF5DataSet(datasetFile, Context.NEIGHBORS)
    d: int = len(xq[0])
    logging.info(f"Dimensions: {d}")
    logging.info(f"Dataset size: {len(xq)}")
    return d, xq, gt


def runSearch(index: faiss.Index, d: int, xq: np.ndarray, gt: HDF5DataSet):
    logging.info(f"Dimensions : {d}")

    hnswParameters = faiss.SearchParametersHNSW()
    hnswParameters.efSearch = 512

    D, I = index.search(xq, 100, params=hnswParameters)

    logging.info(f"Recall at 100 : is {recall_at_r(I, gt, 100, 100, len(xq))}")
    logging.info(f"Recall at 1 : is {recall_at_r(I, gt, 1, 1, len(xq))}")


def main(argv):
    opts, args = getopt.getopt(argv, "", ["graph_input_file=", "dataset_file="])
    graphInputFile = None
    datasetFile = None
    for opt, arg in opts:
        if opt == '-h':
            print('--graph_input_file <inputfile>')
            sys.exit()
        elif opt in "--graph_input_file":
            graphInputFile = arg
        elif opt in "--dataset_file":
            datasetFile = arg
    d, xq, gt = prepare_data(datasetFile=datasetFile)
    index = loadGraphFromFile(graphInputFile)
    runSearch(index, d, xq, gt)


if __name__ == "__main__":
    main(sys.argv[1:])
