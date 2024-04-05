import getopt
from enum import Enum

import numpy as np
import sys
from python.utils.dataset import HDF5DataSet, Context
from python.utils.timer import timer_func


class IndexTypes(Enum):
    CPU = 1

    GPU = 2


@timer_func
def prepare_data(datasetFile: str):
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)
    xb: np.ndarray = index_dataset.read(index_dataset.size())
    d: int = len(xb[0])
    print(f"Dimensions: {d}")
    print(f"Dataset size: {len(xb)}")
    ids = [i for i in range(len(xb))]
    return d, xb, ids


def indexData(datasetFile: str, indexType: IndexTypes):
    d, xb, ids = prepare_data(datasetFile)
    if indexType == IndexTypes.CPU:
        from python.cpu.create_cpu_index import indexData as indexDataInCpu
        indexDataInCpu(d, xb, ids)

    if indexType == IndexTypes.GPU:
        from python.gpu.create_gpu_index import indexData as indexDataInGpu
        indexDataInGpu(d, xb, ids)


def main(argv):
    opts, args = getopt.getopt(argv, "", ["dataset_file=", "index_type="])
    datasetFile = None
    indexType = None
    for opt, arg in opts:
        if opt == '-h':
            print('--dataset_file <dataset file path>')
            print(f'--index_type should have a value {IndexTypes}')
            sys.exit()
        elif opt in "--dataset_file":
            datasetFile = arg
        elif opt == '--index_type':
            if IndexTypes.__contains__(arg):
                indexType = IndexTypes.__getitem__(arg)
            else:
                print(f"Not valid Index Type. Please select a value from {IndexTypes}")

    indexData(datasetFile, indexType)


if __name__ == "__main__":
    main(sys.argv[1:])
