import logging

import faiss
from python.decorators.timer import timer_func
import os
import math


def indexData(d, xb, ids, file_to_write="cpuIndex.hnsw.graph", ef_construction:int = 512, ef_search:int = 512):
    num_of_parallel_threads = min(math.floor(os.cpu_count()/4), 1)
    logging.info(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    cpuPureHNSWIndex: faiss.IndexHNSWFlat = faiss.index_factory(d, "HNSW16,Flat", faiss.METRIC_L2)

    cpuPureHNSWIndex.hnsw.ef_search = ef_search
    cpuPureHNSWIndex.hnsw.ef_construction = ef_construction

    cpuIdMapIndex = faiss.IndexIDMap(cpuPureHNSWIndex)

    @timer_func
    def indexDataInIndex(index: faiss.Index, ids, xb):
        index.add_with_ids(xb, ids)

    print("Creating CPU Index..")
    indexDataInIndex(cpuIdMapIndex, ids, xb)
    print("Writing CPU Index...")

    @timer_func
    def writeIndex(index, fileName):
        faiss.write_index(index, fileName)

    writeIndex(cpuIdMapIndex, file_to_write)
