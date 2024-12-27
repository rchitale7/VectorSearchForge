import sys
sys.path.append('/tmp/faiss/build/faiss/python')
import faiss
import logging
from timeit import default_timer as timer
import math
import numpy as np
import time

from numpy import dtype


def indexData(d:int, xb:np.ndarray, ids:np.ndarray, indexingParams:dict, space_type:str, file_to_write:str="gpuIndex.cagra.graph"):
    num_of_parallel_threads = 8
    logging.info(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    res = faiss.StandardGpuResources()
    metric = faiss.METRIC_L2
    if space_type == "innerproduct":
        metric = faiss.METRIC_INNER_PRODUCT
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 64 if indexingParams.get('intermediate_graph_degree') is None else indexingParams['intermediate_graph_degree']
    cagraIndexConfig.graph_degree = 32 if indexingParams.get('graph_degree') == None else indexingParams['graph_degree']
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    #cagraIndexConfig.conservative_memory_allocation = True
    #cagraIndexConfig.store_dataset = False

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 10 if indexingParams.get('kmeans_n_iters') == None else indexingParams['kmeans_n_iters']
    cagraIndexIVFPQConfig.pq_bits = 8 if indexingParams.get('pq_bits') == None else indexingParams['pq_bits']
    cagraIndexIVFPQConfig.pq_dim = 32 if indexingParams.get('pq_dim') == None else indexingParams['pq_dim']
    cagraIndexIVFPQConfig.n_lists = int(math.sqrt(len(xb))) if indexingParams.get('n_lists') == None else indexingParams['n_lists']
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 10 if indexingParams.get('kmeans_trainset_fraction') == None else indexingParams['kmeans_trainset_fraction']
    cagraIndexIVFPQConfig.conservative_memory_allocation = True
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 30 if indexingParams.get('n_probes') == None else indexingParams['n_probes']
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    print("Creating GPU Index.. with IVF_PQ")
    t1 = timer()
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    t2 = timer()
    print(f"Indexing time: {t2 - t1} sec")

def indexDataInIndex(index: faiss.Index, ids, xb):
    if ids is None:
        index.train(xb)
    else:
        index.add_with_ids(xb, ids)

if __name__ == "__main__":
    d = 768
    np.random.seed(1234)
    # Create a new memory-mapped array
    shape = (4_000_000, 768)  # Example shape
    filename = 'mmap.npy'

    # mmap_array = np.lib.format.open_memmap(filename, mode='w+', dtype='float32', shape=shape)
    # mmap_array[:] = np.random.rand(*shape)
    # del mmap_array
    print("file is written")

    #xb = np.lib.format.open_memmap(filename, mode='r')
    xb = np.load(file = filename)
    #xb2 = np.load(file=filename)
    #xb = np.vstack((xb, xb2))

    # array1 = np.load(file = filename)
    # array2 = np.load(file = filename)
    # xb = np.vstack((array1, array2))
    print(xb.shape)

    # Save changes to disk
    #xb = np.load('my_array.dat', allow_pickle=True)
    #xb = np.random.random((4_000_000, d)).astype('float32')
    ids = [i for i in range(len(xb))]
    #ids = None
    indexData(d, xb, ids, {}, "l2", "testgpuIndex.cagra.graph")
    print("Indexing done")
    del xb
    time.sleep(10)
    print("Deleted xb")
