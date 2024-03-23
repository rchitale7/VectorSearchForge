import faiss
import getopt
import numpy as np
import time
import sys
from python.utils.dataset import HDF5DataSet, Context
from python.utils.timer import timer_func


@timer_func
def prepare_data(datasetFile: str):
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)
    xb: np.ndarray = index_dataset.read(index_dataset.size())
    d:int = len(xb[0])
    print(f"Dimensions: {d}")
    print(f"Dataset size: {len(xb)}")
    ids = [i for i in range(len(xb))]
    return d, xb, ids


def indexData(datasetFile: str):
    d, xb, ids = prepare_data(datasetFile)

    cpuPureHNSWIndex: faiss.IndexHNSWFlat = faiss.index_factory(d, "HNSW16,Flat", faiss.METRIC_L2)

    cpuPureHNSWIndex.hnsw.ef_search = 100
    cpuPureHNSWIndex.hnsw.ef_construction = 100

    cpuIdMapIndex = faiss.IndexIDMap(cpuPureHNSWIndex)

    print("Creating CPU Index..")
    indexDataInIndex(cpuIdMapIndex, ids, xb)
    print("Writing CPU Index...")

    @timer_func
    def writeIndex(index, fileName):
        faiss.write_index(index, fileName)

    writeIndex(cpuIdMapIndex, "cpuIndex.hnsw.graph")

    res = faiss.StandardGpuResources()

    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 8
    cagraIndexConfig.graph_degree = 4
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    cagraIndexConfig.build_algo = faiss.graph_build_algo_NN_DESCENT

    cagraIndex = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    cagraIndex_withoutIds = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    idMapIndex = faiss.IndexIDMap(cagraIndex)

    indexData_withoutIds(cagraIndex_withoutIds, xb)

    indexDataInIndex(idMapIndex, ids, xb)
    writeCagraIndexOnFile(idMapIndex, cagraIndex, "sift.cagra.graph")

@timer_func
def indexData_withoutIds(index:faiss.GpuIndexCagra, xb):
    index.train(xb)


@timer_func
def indexDataInIndex(index: faiss.Index, ids, xb):
    index.add_with_ids(xb, ids)


@timer_func
def writeCagraIndexOnFile(idMapIndex: faiss.Index, cagraIndex: faiss.GpuIndexCagra, outputFileName: str):
    cpuIndex = faiss.IndexHNSWCagra()
    cagraIndex.copyTo(cpuIndex)
    idMapIndex.index = cpuIndex
    faiss.write_index(idMapIndex, outputFileName)


def main(argv):
    opts, args = getopt.getopt(argv, "", ["dataset_file="])
    datasetFile = None
    for opt, arg in opts:
        if opt == '-h':
            print('--dataset_file <dataset file path>')
            sys.exit()
        elif opt in "--dataset_file":
            datasetFile = arg

    indexData(datasetFile)


if __name__ == "__main__":
    main(sys.argv[1:])
