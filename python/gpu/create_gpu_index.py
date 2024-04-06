import faiss
from python.utils.timer import timer_func
import os


def indexData(d, xb, ids):
    faiss.omp_set_num_threads(os.cpu_count() - 1)
    res = faiss.StandardGpuResources()

    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 32
    cagraIndexConfig.nn_descent_niter = 10
    cagraIndexConfig.graph_degree = 16
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    cagraIndexConfig.build_algo = faiss.graph_build_algo_NN_DESCENT

    print("Creating GPU Index.. with NN DESCENT")
    cagraIndex = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    idMapIndex = faiss.IndexIDMap(cagraIndex)

    indexDataInIndex(idMapIndex, ids, xb)
    print("Writing GPU Index.. with NN DESCENT")
    writeCagraIndexOnFile(idMapIndex, cagraIndex, "siftNN_DESCENT.cagra.graph")

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 10
    cagraIndexIVFPQConfig.pq_bits = 4
    cagraIndexIVFPQConfig.pq_dim = 32
    cagraIndexIVFPQConfig.n_lists = 1000
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 10
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 30
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    print("Creating GPU Index.. with IVF_PQ")
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, faiss.METRIC_L2, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)

    print("Creating GPU Index.. with IVF_PQ")
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    writeCagraIndexOnFile(idMapIVFPQIndex, cagraIVFPQIndex, "siftIVF_PQ.cagra.graph")


@timer_func
def indexDataInIndex(index: faiss.Index, ids, xb):
    index.add_with_ids(xb, ids)


@timer_func
def writeCagraIndexOnFile(idMapIndex: faiss.Index, cagraIndex: faiss.GpuIndexCagra, outputFileName: str):
    cpuIndex = faiss.IndexHNSWCagra()
    cagraIndex.copyTo(cpuIndex)
    idMapIndex.index = cpuIndex
    faiss.write_index(idMapIndex, outputFileName)
