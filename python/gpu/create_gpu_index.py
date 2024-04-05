import faiss
from python.utils.timer import timer_func


def indexData(d, xb, ids):
    res = faiss.StandardGpuResources()

    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 32
    cagraIndexConfig.graph_degree = 16
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    cagraIndexConfig.build_algo = faiss.graph_build_algo_NN_DESCENT

    print("Creating GPU Index.. with NN DESCENT")
    cagraIndex = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    idMapIndex = faiss.IndexIDMap(cagraIndex)

    indexDataInIndex(idMapIndex, ids, xb)
    print("Writing GPU Index.. with NN DESCENT")
    writeCagraIndexOnFile(idMapIndex, cagraIndex, "gistNN_DESCENT.cagra.graph")

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    print("Creating GPU Index.. with IVF_PQ")
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)
    print("Creating GPU Index.. with IVF_PQ")
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    writeCagraIndexOnFile(idMapIVFPQIndex, cagraIVFPQIndex, "gistIVF_PQ.cagra.graph")


@timer_func
def indexDataInIndex(index: faiss.Index, ids, xb):
    index.add_with_ids(xb, ids)


@timer_func
def writeCagraIndexOnFile(idMapIndex: faiss.Index, cagraIndex: faiss.GpuIndexCagra, outputFileName: str):
    cpuIndex = faiss.IndexHNSWCagra()
    cagraIndex.copyTo(cpuIndex)
    idMapIndex.index = cpuIndex
    faiss.write_index(idMapIndex, outputFileName)
