import logging

import faiss
from python.decorators.timer import timer_func
from python.utils.common_utils import get_omp_num_threads
from timeit import default_timer as timer


def indexData(d, xb, ids, file_to_write="cpuIndex.hnsw.graph", ef_construction:int = 512, ef_search:int = 512) -> dict:
    num_of_parallel_threads = get_omp_num_threads()
    logging.info(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    cpuPureHNSWIndex: faiss.IndexHNSWFlat = faiss.index_factory(d, "HNSW16,Flat", faiss.METRIC_L2)

    cpuPureHNSWIndex.hnsw.ef_search = ef_search
    cpuPureHNSWIndex.hnsw.ef_construction = ef_construction

    cpuIdMapIndex = faiss.IndexIDMap(cpuPureHNSWIndex)

    @timer_func
    def indexDataInIndex(index: faiss.Index, ids, xb):
        index.add_with_ids(xb, ids)
    t1 = timer()
    indexDataInIndex(cpuIdMapIndex, ids, xb)
    t2 = timer()
    indexTime = t2 - t1
    @timer_func
    def writeIndex(index, fileName):
        faiss.write_index(index, fileName)
    t1 = timer()
    writeIndex(cpuIdMapIndex, file_to_write)
    t2 = timer()
    writeIndexTime = t2 - t1
    return {"indexTime": indexTime, "writeIndexTime": writeIndexTime, "totalTime": indexTime + writeIndexTime, "unit": "seconds" }
