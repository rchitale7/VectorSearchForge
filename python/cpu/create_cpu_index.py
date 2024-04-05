import faiss
from python.utils.timer import timer_func
import os


def indexData(d, xb, ids):
    faiss.omp_set_num_threads(os.cpu_count() - 1)
    cpuPureHNSWIndex: faiss.IndexHNSWFlat = faiss.index_factory(d, "HNSW16,Flat", faiss.METRIC_L2)

    cpuPureHNSWIndex.hnsw.ef_search = 100
    cpuPureHNSWIndex.hnsw.ef_construction = 100

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

    writeIndex(cpuIdMapIndex, "cpuIndex.hnsw.graph")
